"""Build the Galton board scene in Blender (bpy module) and render frames.

usage: python scene.py SIM.npz OUT_DIR [--frames A-B] [--samples N]
                       [--height H] [--save-blend FILE]
"""
import argparse
import colorsys
import math
import os
import sys
import time

import bpy  # noqa: I001  (bpy must be imported before bmesh)
import bmesh
import numpy as np
from bpy_extras import anim_utils
from mathutils import Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B  # noqa: E402

CM = 0.01


# ---------------------------------------------------------------- helpers

def srgb_to_linear(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def link(obj, parent=None):
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def box_mesh(name, sx, sy, sz):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy
        v.co.z *= sz
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def add_box(name, lo, hi, mat, parent, bevel=0.06, rot_y=0.0, center=None):
    size = [h - l for l, h in zip(lo, hi)]
    obj = bpy.data.objects.new(name, box_mesh(name, *size))
    obj.location = center if center is not None else [(l + h) / 2 for l, h in zip(lo, hi)]
    obj.rotation_euler = (0, rot_y, 0)
    obj.data.materials.append(mat)
    if bevel > 0:
        m = obj.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 3
        m.limit_method = "ANGLE"
    return link(obj, parent)


def cylinder_mesh(name, radius, depth, segments=64, bevel=0.0):
    """Cylinder along Y from y=0 to y=depth, smooth sides, bevelled front."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius,
                          radius2=radius, depth=depth)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                     matrix=Euler((math.pi / 2, 0, 0)).to_matrix())
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, depth / 2, 0))
    for f in bm.faces:
        f.smooth = len(f.verts) == 4
    if bevel > 0:
        front = [e for e in bm.edges
                 if all(abs(v.co.y) < 1e-5 for v in e.verts) and len(e.link_faces) == 2]
        res = bmesh.ops.bevel(bm, geom=front, offset=bevel, segments=6, profile=0.5,
                              affect="EDGES")
        for f in res["faces"]:
            f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def sphere_mesh(name, radius):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=5, radius=radius)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


# -------------------------------------------------------------- materials

def node_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return mat, nt, out


def principled(nt, **inputs):
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    for k, v in inputs.items():
        bsdf.inputs[k].default_value = v
    return bsdf


def walnut_material(name, grain_axis):
    """Lacquered walnut; grain_axis is 'Z' (vertical) or 'X' (horizontal)."""
    mat, nt, out = node_mat(name)
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    stretch = (1.0, 1.0, 0.07) if grain_axis == "Z" else (0.07, 1.0, 1.0)
    mapping.inputs["Scale"].default_value = stretch
    L.new(tc.outputs["Object"], mapping.inputs["Vector"])

    warp = nt.nodes.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 0.35
    warp.inputs["Detail"].default_value = 4.0
    L.new(mapping.outputs["Vector"], warp.inputs["Vector"])
    mix_co = nt.nodes.new("ShaderNodeMix")
    mix_co.data_type = "VECTOR"
    mix_co.inputs["Factor"].default_value = 0.35
    L.new(mapping.outputs["Vector"], mix_co.inputs[4])
    L.new(warp.outputs["Color"], mix_co.inputs[5])

    rings = nt.nodes.new("ShaderNodeTexWave")
    rings.wave_type = "RINGS"
    rings.rings_direction = "Z" if grain_axis == "Z" else "X"
    rings.inputs["Scale"].default_value = 0.45
    rings.inputs["Distortion"].default_value = 9.0
    rings.inputs["Detail"].default_value = 4.0
    rings.inputs["Detail Scale"].default_value = 1.2
    L.new(mix_co.outputs[1], rings.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.1, (0.030, 0.014, 0.007, 1)
    els[1].position, els[1].color = 1.0, (0.095, 0.047, 0.023, 1)
    mid = els.new(0.55)
    mid.color = (0.058, 0.028, 0.013, 1)
    L.new(rings.outputs["Fac"], ramp.inputs["Fac"])

    # fine open pores, stretched hard along the grain
    pores_map = nt.nodes.new("ShaderNodeMapping")
    pstretch = (1.0, 1.0, 0.02) if grain_axis == "Z" else (0.02, 1.0, 1.0)
    pores_map.inputs["Scale"].default_value = [s * 40 for s in pstretch]
    L.new(tc.outputs["Object"], pores_map.inputs["Vector"])
    pores = nt.nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 1.0
    pores.inputs["Detail"].default_value = 2.0
    L.new(pores_map.outputs["Vector"], pores.inputs["Vector"])
    pore_ramp = nt.nodes.new("ShaderNodeValToRGB")
    pore_ramp.color_ramp.elements[0].position = 0.25
    pore_ramp.color_ramp.elements[0].color = (0.55, 0.55, 0.55, 1)
    pore_ramp.color_ramp.elements[1].position = 0.45
    L.new(pores.outputs["Fac"], pore_ramp.inputs["Fac"])
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], tint.inputs[6])
    L.new(pore_ramp.outputs["Color"], tint.inputs[7])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03
    bump.inputs["Distance"].default_value = 0.02
    L.new(pores.outputs["Fac"], bump.inputs["Height"])

    bsdf = principled(nt, **{"Roughness": 0.42, "Coat Weight": 0.6,
                             "Coat Roughness": 0.08})
    L.new(tint.outputs[2], bsdf.inputs["Base Color"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def smoked_oak_material():
    """Dark, satin-finished oak for the back panel: fine straight grain."""
    mat, nt, out = node_mat("SmokedOak")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    fine_map = nt.nodes.new("ShaderNodeMapping")
    fine_map.inputs["Scale"].default_value = (7.0, 7.0, 0.22)
    L.new(tc.outputs["Object"], fine_map.inputs["Vector"])
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 1.0
    fine.inputs["Detail"].default_value = 3.0
    fine.inputs["Roughness"].default_value = 0.65
    L.new(fine_map.outputs["Vector"], fine.inputs["Vector"])
    broad = nt.nodes.new("ShaderNodeTexNoise")
    broad.inputs["Scale"].default_value = 0.08
    broad.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], broad.inputs["Vector"])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 0.8
    L.new(fine.outputs["Fac"], add.inputs[0])
    L.new(broad.outputs["Fac"], add.inputs[2])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.55, (0.014, 0.0085, 0.0055, 1)
    els[1].position, els[1].color = 1.05, (0.050, 0.030, 0.018, 1)
    L.new(add.outputs[0], ramp.inputs["Fac"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.01
    L.new(fine.outputs["Fac"], bump.inputs["Height"])
    bsdf = principled(nt, **{"Roughness": 0.5, "Coat Weight": 0.25,
                             "Coat Roughness": 0.22})
    L.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def brass_material():
    mat, nt, out = node_mat("Brass")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.11
    rough.inputs["To Max"].default_value = 0.24
    L.new(noise.outputs["Fac"], rough.inputs["Value"])
    bsdf = principled(nt, **{"Base Color": (0.86, 0.62, 0.30, 1), "Metallic": 1.0})
    L.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def ball_material():
    """Glossy lacquered marble; colour comes from the object colour."""
    mat, nt, out = node_mat("Ball")
    L = nt.links
    info = nt.nodes.new("ShaderNodeObjectInfo")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # faint surface wear in object space so spin is visible in the highlights
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.35
    rough.inputs["From Max"].default_value = 0.75
    rough.inputs["To Min"].default_value = 0.02
    rough.inputs["To Max"].default_value = 0.16
    L.new(noise.outputs["Fac"], rough.inputs["Value"])
    bsdf = principled(nt, **{"Roughness": 0.22, "Coat Weight": 1.0, "Coat IOR": 1.55})
    L.new(info.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(rough.outputs["Result"], bsdf.inputs["Coat Roughness"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def glass_material():
    """Thin glass pane: Fresnel-weighted reflection over a straight-through
    transparent layer. A 3 mm pane barely offsets what is behind it, so this
    looks like real glass while costing far less than refraction."""
    mat, nt, out = node_mat("Glass")
    L = nt.links
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.93, 0.95, 0.94, 1)
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.02
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.52
    mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(fres.outputs[0], mix.inputs["Fac"])
    L.new(transparent.outputs[0], mix.inputs[1])
    L.new(gloss.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def simple_material(name, color, rough, spec=0.5):
    mat, nt, out = node_mat(name)
    bsdf = principled(nt, **{"Base Color": color, "Roughness": rough,
                             "Specular IOR Level": spec})
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# ------------------------------------------------------------------ scene

def build_board(root, mats):
    walnut_v, walnut_h, brass, glass = (mats["walnut_v"], mats["walnut_h"],
                                        mats["brass"], mats["glass"])
    hw, gap = B.HALF_W, B.GAP
    y_front = -gap / 2 - B.GLASS_T
    y_back = gap / 2 + B.BACK_T
    top = B.BOARD_TOP_Z + B.TOP_RAIL_H
    outer = hw + B.FRAME_W
    post_front = y_front - 0.5

    add_box("BackPanel", (-outer + 0.2, gap / 2, -0.2), (outer - 0.2, y_back, top - 0.2),
            mats["back"], root, bevel=0.0)
    # single front face of the pane, seen by camera rays only: shadow and
    # bounce rays skip it, which keeps it nearly free to render
    pane = bpy.data.meshes.new("Glass")
    x0, x1, z1 = -hw - 0.4, hw + 0.4, top - 0.05
    pane.from_pydata([(x0, y_front, 0), (x1, y_front, 0), (x1, y_front, z1), (x0, y_front, z1)],
                     [], [(0, 1, 2, 3)])
    pane.materials.append(glass)
    gobj = link(bpy.data.objects.new("Glass", pane), root)
    gobj.visible_shadow = False
    gobj.visible_diffuse = False
    gobj.visible_glossy = False
    gobj.visible_transmission = False
    gobj.visible_volume_scatter = False
    for s in (-1, 1):
        add_box(f"Post{s}", (min(s * hw, s * outer), post_front, 0.0),
                (max(s * hw, s * outer), y_back + 0.2, top), walnut_v, root, bevel=0.12)
        lo_x, hi_x = sorted((s * B.TOP_SLOT_HALF, s * (hw + 0.01)))
        add_box(f"TopRail{s}", (lo_x, post_front, B.BOARD_TOP_Z), (hi_x, y_back + 0.2, top),
                walnut_h, root, bevel=0.12)
    add_box("BinFloor", (-hw, post_front, -0.6), (hw, y_back, 0.0), walnut_h, root, bevel=0.05)
    add_box("Base", (-B.BASE_W / 2, -B.BASE_D / 2, -B.BASE_H), (B.BASE_W / 2, B.BASE_D / 2, -0.6),
            walnut_h, root, bevel=0.25)

    peg_me = cylinder_mesh("Peg", B.PEG_R, gap + 0.6, bevel=0.16)
    peg_me.materials.append(brass)
    for i, (x, z) in enumerate(B.peg_positions()):
        o = bpy.data.objects.new(f"Peg{i}", peg_me)
        o.location = (x, -gap / 2, z)
        link(o, root)

    wedge_me = box_mesh("Wedge", 2 * B.WEDGE_HALF, gap, 2 * B.WEDGE_HALF)
    wedge_me.materials.append(brass)
    for i, (x, z) in enumerate(B.wedge_positions()):
        o = bpy.data.objects.new(f"Wedge{i}", wedge_me)
        o.location = (x, 0.0, z)
        o.rotation_euler = (0, math.pi / 4, 0)
        m = o.modifiers.new("bevel", "BEVEL")
        m.width, m.segments = 0.05, 3
        link(o, root)

    cap_me = cylinder_mesh("DividerCap", B.DIVIDER_T / 2, gap)
    cap_me.materials.append(brass)
    for i, x in enumerate(B.divider_positions()):
        add_box(f"Divider{i}", (x - B.DIVIDER_T / 2, -gap / 2, 0.0),
                (x + B.DIVIDER_T / 2, gap / 2, B.DIVIDER_H), brass, root, bevel=0.0)
        o = bpy.data.objects.new(f"DividerCap{i}", cap_me)
        o.location = (x, -gap / 2, B.DIVIDER_H)
        link(o, root)

    for i, (cx, cz, length, ang) in enumerate(B.funnel_walls()):
        o = add_box(f"Funnel{i}", (0, 0, 0), (length + 0.2, gap, B.FUNNEL_T), walnut_h, root,
                    bevel=0.05, rot_y=-ang, center=(cx, 0.0, cz))

    # studio surroundings
    add_box("Table", (-400, -300, -B.BASE_H - 2), (400, 300, -B.BASE_H), mats["table"], root,
            bevel=0.0)
    add_box("Backdrop", (-600, 140, -B.BASE_H - 2), (600, 142, 500), mats["backdrop"], root,
            bevel=0.0)


def build_balls(root, mat, sim):
    pos, quat = sim["pos"], sim["quat"]
    hz = int(sim["sample_hz"])
    n = pos.shape[1]
    me = sphere_mesh("Ball", B.BALL_R)
    me.materials.append(mat)
    fps = bpy.context.scene.render.fps
    frames = 1 + np.arange(pos.shape[0]) * fps / hz
    for i in range(n):
        o = bpy.data.objects.new(f"Ball{i:03d}", me)
        hue = 0.0 + 0.80 * i / max(n - 1, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.82, 0.92)
        o.color = (*srgb_to_linear(rgb), 1.0)
        o.rotation_mode = "QUATERNION"
        link(o, root)
        o.keyframe_insert("location", frame=1)
        o.keyframe_insert("rotation_quaternion", frame=1)
        bag = anim_utils.animdata_get_channelbag_for_assigned_slot(o.animation_data)
        q = quat[:, i, [3, 0, 1, 2]]          # pybullet xyzw -> blender wxyz
        # flip signs so consecutive keys stay in one hemisphere (no wild spins)
        flips = np.where(np.sum(q[1:] * q[:-1], 1) < 0, -1.0, 1.0)
        q = q * np.concatenate([[1.0], np.cumprod(flips)])[:, None]
        for fc in bag.fcurves:
            data = pos[:, i, :] if fc.data_path == "location" else q
            kp = fc.keyframe_points
            kp.clear()
            kp.add(len(frames))
            co = np.empty(2 * len(frames), np.float32)
            co[0::2] = frames
            co[1::2] = data[:, fc.array_index]
            kp.foreach_set("co", co)
            kp.foreach_set("interpolation", np.full(len(frames), _LINEAR, np.int32))
            fc.update()


_LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value


# ---------------------------------------------------------- camera + light

def smoothstep(a, b, t):
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def camera_state(t):
    """(target_xz_cm, distance_cm, yaw_deg, pitch_deg, fstop) at time t."""
    def lerp(a, b, u):
        return a + (b - a) * u
    u0 = smoothstep(0.0, 3.5, t)        # slow creep on the opening close-up
    u1 = smoothstep(3.0, 14.5, t)       # pull back to reveal the board
    u2 = smoothstep(14.5, 23.0, t)      # gentle settle on the bell curve
    tz = lerp(lerp(28.0, 27.2, u0), 17.0, u1)
    tz = lerp(tz, 16.4, u2)
    dist = lerp(lerp(31.0, 29.5, u0), 79.0, u1)
    dist = lerp(dist, 74.0, u2)
    yaw = lerp(-12.0, -7.0, u1)
    yaw = lerp(yaw, -3.5, u2)
    pitch = lerp(5.0, 1.5, u1)
    fstop = lerp(8.0, 9.0, u1)
    return (0.0, tz), dist, yaw, pitch, fstop


def build_camera(scene, root):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 60
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = 36
    cam_data.dof.use_dof = True
    cam_data.clip_start = 0.01
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    target = bpy.data.objects.new("CamTarget", None)
    scene.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam_data.dof.focus_object = target

    fps = scene.render.fps
    for f in range(scene.frame_start, scene.frame_end + 1):
        t = (f - 1) / fps
        (tx, tz), dist, yaw, pitch, fstop = camera_state(t)
        yr, pr = math.radians(yaw), math.radians(pitch)
        tgt = Vector((tx, 0.0, tz)) * CM
        off = Vector((math.sin(yr) * math.cos(pr), -math.cos(yr) * math.cos(pr),
                      math.sin(pr))) * dist * CM
        target.location = tgt
        cam.location = tgt + off
        cam_data.dof.aperture_fstop = fstop
        target.keyframe_insert("location", frame=f)
        cam.keyframe_insert("location", frame=f)
        cam_data.dof.keyframe_insert("aperture_fstop", frame=f)


def area_light(name, loc, look_at, size, power, color=(1, 1, 1), shape="RECTANGLE",
               size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y:
        ld.size_y = size_y
    ld.energy = power
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    o.location = loc
    d = Vector(look_at) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(o)
    return o


def build_lights():
    c = (0, 0, 0.17)
    area_light("Key", (-0.55, -0.55, 0.62), c, 0.55, 18, (1.0, 0.96, 0.9), size_y=0.38)
    area_light("Fill", (0.75, -0.45, 0.25), c, 0.6, 4, (0.88, 0.93, 1.0), size_y=0.6)
    area_light("Rim", (0.25, 0.45, 0.75), c, 0.35, 6, (1.0, 0.9, 0.8))
    area_light("Strip", (-0.02, -0.5, 0.95), c, 0.8, 5, (1, 1, 1), size_y=0.06)
    spot = bpy.data.lights.new("BackdropSpot", "SPOT")
    spot.energy = 12
    spot.spot_size = math.radians(55)
    spot.spot_blend = 1.0
    spot.color = (1.0, 0.82, 0.62)
    so = bpy.data.objects.new("BackdropSpot", spot)
    so.location = (0.0, 0.2, 0.15)
    so.rotation_euler = (Vector((0.0, 1.4, 0.35)) - so.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(so)


def setup_render(scene, out_dir, samples, height):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.03
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_quality = "BALANCED"
    cy.max_bounces = 6
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 3
    cy.transmission_bounces = 4
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.5
    cy.sample_clamp_indirect = 6.0
    cy.use_light_tree = True
    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.35
    scene.render.resolution_x = height * 9 // 16
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = B.FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = os.path.join(out_dir, "f_")
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.012, 0.014, 1)
    scene.world = world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--height", type=int, default=1920,
                    help="render height; width follows 9:16")
    ap.add_argument("--save-blend", default=None)
    args = ap.parse_args()

    t_start = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene, args.out_dir, args.samples, args.height)
    scene.frame_start = 1
    scene.frame_end = int(round(B.DURATION * B.FPS))

    sim = np.load(args.sim)
    root = bpy.data.objects.new("BoardRoot", None)
    root.scale = (CM, CM, CM)
    scene.collection.objects.link(root)
    mats = {
        "walnut_v": walnut_material("WalnutV", "Z"),
        "walnut_h": walnut_material("WalnutH", "X"),
        "brass": brass_material(),
        "back": smoked_oak_material(),
        "glass": glass_material(),
        "table": simple_material("Table", (0.018, 0.017, 0.016, 1), 0.38, 0.4),
        "backdrop": simple_material("Backdrop", (0.05, 0.048, 0.046, 1), 0.9, 0.1),
    }
    build_board(root, mats)
    build_balls(root, ball_material(), sim)
    build_camera(scene, root)
    build_lights()
    print(f"scene built in {time.time() - t_start:.1f}s", flush=True)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))

    if args.frames:
        a, _, b = args.frames.partition("-")
        frames = range(int(a), int(b or a) + 1)
    else:
        frames = range(scene.frame_start, scene.frame_end + 1)
    os.makedirs(args.out_dir, exist_ok=True)
    for f in frames:
        path = os.path.join(args.out_dir, f"f_{f:04d}.png")
        if os.path.exists(path):
            continue
        scene.frame_set(f)
        scene.render.filepath = path
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {path} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
