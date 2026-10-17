"""Shared geometry for the Galton board.

All dimensions are in centimetres, z is up, the board lies in the XZ plane
and the camera looks along +Y. Both the physics sim and the Blender scene
import this module so colliders and visible meshes always match.
"""
import math

FPS = 30
DURATION = 23.0                 # seconds of finished video

BALL_R = 0.30                   # 6 mm marbles
BALL_MASS = 0.28                # grams (glass, 2.5 g/cm^3)

PEG_R = 0.40
PEG_SPACING = 1.8               # horizontal spacing == bin width
PEG_ROWS = 11
ROW_DZ = PEG_SPACING * math.sqrt(3) / 2

HALF_W = 5 * PEG_SPACING        # inner half width (10 bins)
GAP = 0.70                      # glass-to-back clearance (ball is 0.60)
GLASS_T = 0.30
BACK_T = 0.60

DIVIDER_T = 0.14
DIVIDER_H = 11.5
LAST_ROW_Z = DIVIDER_H + 1.2
TOP_ROW_Z = LAST_ROW_Z + (PEG_ROWS - 1) * ROW_DZ

FUNNEL_EXIT_Z = TOP_ROW_Z + 1.25
FUNNEL_APERTURE = 1.6           # ~2.7 ball diameters
FUNNEL_TOP_HALF = 4.2
FUNNEL_ANGLE = math.radians(48)
FUNNEL_TOP_Z = FUNNEL_EXIT_Z + (FUNNEL_TOP_HALF - FUNNEL_APERTURE / 2) * math.tan(FUNNEL_ANGLE)
FUNNEL_T = 0.35

BOARD_TOP_Z = FUNNEL_TOP_Z + 1.0
FRAME_W = 1.4                   # walnut side posts
FRAME_D = 2.4                   # frame depth (y)
TOP_RAIL_H = 1.3
TOP_SLOT_HALF = 3.2             # opening in the top rail
BASE_H = 3.0
BASE_W = 2 * (HALF_W + FRAME_W) + 3.0
BASE_D = 7.0

SPAWN_Z = BOARD_TOP_Z + 10.0


def peg_positions():
    """(x, z) of every peg: even rows have a peg on the centreline."""
    pegs = []
    for i in range(PEG_ROWS):
        z = TOP_ROW_Z - i * ROW_DZ
        if i % 2 == 0:
            xs = [k * PEG_SPACING for k in range(-4, 5)]
        else:
            # no outermost pegs: they would leave a ball-trapping gap at the wall
            xs = [(k + 0.5) * PEG_SPACING for k in range(-4, 4)]
        pegs += [(x, z) for x in xs]
    return pegs


WEDGE_HALF = 0.45                # 45-degree deflectors on the side walls


def wedge_positions():
    """(x, z) centres of the square wedges on the walls at every even row.

    Each is a square rotated 45 degrees about y and centred on the wall, so
    its inner half is a triangle whose sloped faces push balls back inward.
    """
    out = []
    for i in range(0, PEG_ROWS, 2):
        z = TOP_ROW_Z - i * ROW_DZ
        out += [(-HALF_W, z), (HALF_W, z)]
    return out


def divider_positions():
    return [k * PEG_SPACING for k in range(-4, 5)]


def funnel_walls():
    """Each funnel wall as (centre_x, centre_z, length, angle_about_y).

    The wall's inner face runs from the aperture edge up to the top edge;
    the slab is offset outward by half its thickness so the inner face is
    exactly on that line.
    """
    walls = []
    for side in (-1, 1):
        x0, z0 = side * FUNNEL_APERTURE / 2, FUNNEL_EXIT_Z
        x1, z1 = side * FUNNEL_TOP_HALF, FUNNEL_TOP_Z
        length = math.hypot(x1 - x0, z1 - z0)
        ang = math.atan2(z1 - z0, x1 - x0)
        # outward normal of the inner face (points away from the opening)
        nx, nz = math.sin(ang), -math.cos(ang)
        if side < 0:
            nx, nz = -nx, -nz
        cx = (x0 + x1) / 2 + nx * FUNNEL_T / 2
        cz = (z0 + z1) / 2 + nz * FUNNEL_T / 2
        walls.append((cx, cz, length, ang))
    return walls
