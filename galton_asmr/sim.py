"""Rigid-body simulation of the Galton board with pybullet.

Writes sim.npz containing ball trajectories sampled at SAMPLE_HZ and a list
of collision onsets (time, kind, impact speed cm/s, x, z) for the audio.
"""
import math
import os
import sys

import numpy as np
import pybullet as p

import board as B

DT = 1.0 / 1000.0
SAMPLE_HZ = 60
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
FAST = "--fast" in sys.argv          # skip contact logging (layout tuning)
PREROLL = 1.4                        # seconds simulated before the video starts
BALL_E = float(os.environ.get("BALL_E", 0.5))
PEG_E = float(os.environ.get("PEG_E", 0.5))
OUT = sys.argv[1] if len(sys.argv) > 1 else "sim.npz"

KIND_BALL, KIND_PEG, KIND_DIVIDER, KIND_WOOD, KIND_GLASS, KIND_BACK = range(6)
ZERO = (0.0, 0.0, 0.0)


def spawn_times(rng):
    """Already pouring when the video starts, a steady rain, a taper, and a
    few last single balls for the ending."""
    times = []
    t = -PREROLL
    while t < 17.0:
        if t < 4.0:
            rate = 5.0 + (t + PREROLL) / (4.0 + PREROLL) * 8.0
        elif t < 13.5:
            rate = 13.0
        else:
            rate = 13.0 * (17.0 - t) / 3.5 + 1.0
        times.append(t)
        t += rng.exponential(1.0 / rate) * 0.5 + 0.5 / rate
    times += [18.0, 19.0, 20.2]
    return np.array(times)


def box(half, pos, ang_y=0.0, kind=KIND_WOOD, rest=0.55, fric=0.4, kinds=None):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    orn = p.getQuaternionFromEuler((0, ang_y, 0))
    bid = p.createMultiBody(0, col, -1, pos, orn)
    p.changeDynamics(bid, -1, restitution=rest, lateralFriction=fric)
    kinds[bid] = kind
    return bid


def build_world(kinds):
    y_mid = 0.0
    depth = B.GAP / 2
    # back panel and front glass (low friction so balls do not roll on them)
    box((B.HALF_W + 1, B.BACK_T / 2, 30), (0, depth + B.BACK_T / 2, 20),
        kind=KIND_BACK, rest=0.4, fric=0.05, kinds=kinds)
    box((B.HALF_W + 1, B.GLASS_T / 2, 30), (0, -depth - B.GLASS_T / 2, 20),
        kind=KIND_GLASS, rest=0.4, fric=0.05, kinds=kinds)
    # floor and side walls
    box((B.HALF_W + 1, 2, 0.5), (0, y_mid, -0.5), kinds=kinds)
    for s in (-1, 1):
        box((0.5, 2, 30), (s * (B.HALF_W + 0.5), y_mid, 25), kinds=kinds)
    # bin dividers (brass) with a rounded top cap
    for x in B.divider_positions():
        box((B.DIVIDER_T / 2, depth, B.DIVIDER_H / 2), (x, y_mid, B.DIVIDER_H / 2),
            kind=KIND_DIVIDER, rest=PEG_E, fric=0.25, kinds=kinds)
        cap = p.createCollisionShape(p.GEOM_CYLINDER, radius=B.DIVIDER_T / 2, height=2 * depth)
        cid = p.createMultiBody(0, cap, -1, (x, y_mid, B.DIVIDER_H),
                                p.getQuaternionFromEuler((math.pi / 2, 0, 0)))
        p.changeDynamics(cid, -1, restitution=PEG_E, lateralFriction=0.25)
        kinds[cid] = KIND_DIVIDER
    # pegs
    peg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=B.PEG_R, height=2 * depth + 0.4)
    orn = p.getQuaternionFromEuler((math.pi / 2, 0, 0))
    for x, z in B.peg_positions():
        pid = p.createMultiBody(0, peg_col, -1, (x, y_mid, z), orn)
        p.changeDynamics(pid, -1, restitution=PEG_E, lateralFriction=0.25)
        kinds[pid] = KIND_PEG
    # wall deflector wedges
    for x, z in B.wedge_positions():
        box((B.WEDGE_HALF, depth, B.WEDGE_HALF), (x, y_mid, z), ang_y=math.pi / 4,
            kind=KIND_PEG, rest=PEG_E, fric=0.25, kinds=kinds)
    # funnel
    for cx, cz, length, ang in B.funnel_walls():
        box((length / 2 + 0.1, depth, B.FUNNEL_T / 2), (cx, y_mid, cz), ang_y=-ang,
            kind=KIND_WOOD, rest=0.5, fric=0.35, kinds=kinds)


def main():
    rng = np.random.default_rng(SEED)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -981.0)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=40,
                                restitutionVelocityThreshold=2.0,
                                contactBreakingThreshold=0.01)
    kinds = {}
    build_world(kinds)

    t_spawn = spawn_times(rng)
    n = len(t_spawn)
    spawn_x = np.clip(rng.normal(0, 0.55, n), -1.6, 1.6)
    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=B.BALL_R)

    pre_steps = int(round(PREROLL / DT))
    n_steps = pre_steps + int(round(B.DURATION / DT))
    every = int(round(1.0 / (SAMPLE_HZ * DT)))
    n_samples = (n_steps - pre_steps) // every + 1
    pos = np.zeros((n_samples, n, 3), np.float32)
    quat = np.zeros((n_samples, n, 4), np.float32)
    quat[..., 3] = 1
    pos[:, :, 0] = spawn_x
    pos[:, :, 2] = B.SPAWN_Z + 25.0          # parked straight above the frame

    balls = [-1] * n
    ball_index = {}
    next_spawn = 0
    vel = {}             # body -> linear velocity after the previous step
    events = []
    prev_pairs = set()

    for step in range(n_steps + 1):
        t = (step - pre_steps) * DT
        while next_spawn < n and t_spawn[next_spawn] <= t:
            i = next_spawn
            bid = p.createMultiBody(B.BALL_MASS, ball_col, -1,
                                    (spawn_x[i], 0.0, B.SPAWN_Z),
                                    p.getQuaternionFromEuler(tuple(rng.uniform(-3, 3, 3))))
            p.changeDynamics(bid, -1, restitution=BALL_E, lateralFriction=0.3,
                             rollingFriction=0.0004, spinningFriction=0.0004,
                             linearDamping=0.02, angularDamping=0.08)
            p.resetBaseVelocity(bid, (rng.normal(0, 4), 0, -rng.uniform(0, 30)),
                                tuple(rng.normal(0, 20, 3)))
            balls[i] = bid
            ball_index[bid] = i
            kinds[bid] = KIND_BALL
            next_spawn += 1

        if step >= pre_steps and (step - pre_steps) % every == 0:
            k = (step - pre_steps) // every
            for i in range(next_spawn):
                xyz, q = p.getBasePositionAndOrientation(balls[i])
                pos[k, i] = xyz
                quat[k, i] = q
        if step == n_steps:
            break

        p.stepSimulation()

        if FAST:
            continue
        # a contact that was not there last step is an impact; its loudness is
        # the approach speed along the contact normal just before the step
        cur = set()
        for c in p.getContactPoints():
            a, b = c[1], c[2]
            key = (a, b) if a < b else (b, a)
            if key in cur:
                continue
            cur.add(key)
            if key in prev_pairs or step < pre_steps:
                continue
            va, vb = vel.get(a, ZERO), vel.get(b, ZERO)
            n_ = c[7]
            speed = abs((va[0] - vb[0]) * n_[0] + (va[1] - vb[1]) * n_[1]
                        + (va[2] - vb[2]) * n_[2])
            ka, kb = kinds.get(a, KIND_WOOD), kinds.get(b, KIND_WOOD)
            kind = kb if ka == KIND_BALL else ka
            events.append(((step - pre_steps) * DT, kind, speed, c[5][0], c[5][2]))
        prev_pairs = cur
        for i in range(next_spawn):
            vel[balls[i]] = p.getBaseVelocity(balls[i])[0]

        if step % 2000 == 0:
            print(f"t={t:5.1f}s balls={next_spawn:3d} events={len(events)}", flush=True)

    ev = np.array(events, np.float64).reshape(-1, 5)
    np.savez_compressed(OUT, pos=pos, quat=quat, t_spawn=t_spawn,
                        sample_hz=SAMPLE_HZ, events=ev)
    final = pos[-1]
    bins = np.histogram(final[:, 0], bins=np.arange(-5, 6) * B.PEG_SPACING)[0]
    print("balls", n, "events", len(ev))
    print("bin counts", bins.tolist(), "max height", float(final[:, 2].max()))


if __name__ == "__main__":
    main()
