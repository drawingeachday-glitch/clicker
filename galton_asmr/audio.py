"""Synthesize the ASMR soundtrack from the simulation's collision events.

Every contact onset logged by sim.py becomes a short modal-synthesis grain:
brass pegs ring with a bright "tink", marbles knock together with a glassy
click, wood gives a soft "tok". Grains are panned by board position, then a
small-room reverb and gentle mastering are applied.

usage: python audio.py SIM.npz OUT.wav
"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

import board as B

SR = 48000
KIND_BALL, KIND_PEG, KIND_DIVIDER, KIND_WOOD, KIND_GLASS, KIND_BACK = range(6)

# per kind: (min audible impact speed cm/s, gain, grain length s)
KIND_PARAMS = {
    KIND_PEG: (4.0, 1.00, 0.12),
    KIND_BALL: (5.0, 0.55, 0.06),
    KIND_DIVIDER: (4.0, 0.70, 0.18),
    KIND_WOOD: (5.0, 0.80, 0.08),
    KIND_GLASS: (6.0, 0.30, 0.06),
    KIND_BACK: (8.0, 0.25, 0.05),
}


def modal(t, freqs, decays, amps, phases):
    out = np.zeros_like(t)
    for f, d, a, ph in zip(freqs, decays, amps, phases):
        if f < 19000:                       # skip partials that would alias
            out += a * np.exp(-t / d) * np.sin(2 * np.pi * f * t + ph)
    return out


def click(n, rng, hp_sos):
    """Sub-millisecond contact transient (filtered noise burst)."""
    m = min(n, int(0.0015 * SR))
    burst = np.zeros(n)
    env = np.exp(-np.arange(m) / (0.00025 * SR))
    burst[:m] = rng.standard_normal(m) * env
    return sosfilt(hp_sos, burst)


def peg_pitch_table():
    rng = np.random.default_rng(1234)
    pegs = np.array(B.peg_positions() + B.wedge_positions())
    # each pin is tuned slightly differently, like real press-fit brass pins
    return pegs, rng.uniform(2300, 4300, len(pegs))


def grain(kind, v, x, z, rng, tables, hp_sos):
    _, gain, length = KIND_PARAMS[kind]
    n = int(length * SR)
    t = np.arange(n) / SR
    hard = np.clip(v / 120.0, 0.0, 1.0)      # harder hits are brighter
    ph = rng.uniform(0, 2 * np.pi, 6)
    if kind == KIND_PEG:
        pegs, f0s = tables
        f0 = f0s[np.argmin((pegs[:, 0] - x) ** 2 + (pegs[:, 1] - z) ** 2)]
        f0 *= rng.uniform(0.995, 1.005)
        ratios = np.array([1.0, 2.71, 5.18])
        # pins pressed into wood are damped: a short ring, mostly "tick"
        decays = np.array([0.028, 0.012, 0.006]) * rng.uniform(0.85, 1.15)
        amps = np.array([1.0, 0.5 + 0.4 * hard, 0.2 + 0.3 * hard])
        body = modal(t, f0 * ratios, decays, amps, ph)
        tr = 0.8 + 0.7 * hard
    elif kind == KIND_BALL:
        f0 = rng.uniform(5200, 7400)
        body = modal(t, f0 * np.array([1.0, 1.53, 2.12]),
                     np.array([0.009, 0.006, 0.004]) * rng.uniform(0.8, 1.2),
                     np.array([1.0, 0.6, 0.35]), ph)
        tr = 0.9 + 0.5 * hard
    elif kind == KIND_DIVIDER:
        f0 = rng.uniform(1650, 1850)
        body = modal(t, f0 * np.array([1.0, 2.76, 5.40, 8.93]),
                     np.array([0.045, 0.02, 0.01, 0.005]),
                     np.array([1.0, 0.6, 0.3 + 0.3 * hard, 0.15]), ph)
        tr = 0.5
    elif kind == KIND_WOOD:
        f0 = rng.uniform(700, 900)
        body = modal(t, f0 * np.array([1.0, 1.9, 3.1, 4.6]),
                     np.array([0.018, 0.011, 0.007, 0.004]),
                     np.array([1.0, 0.7, 0.45, 0.25]), ph)
        tr = 0.7
    elif kind == KIND_GLASS:
        f0 = rng.uniform(3200, 4200)
        body = modal(t, f0 * np.array([1.0, 2.3, 3.9]),
                     np.array([0.02, 0.01, 0.006]), np.array([1.0, 0.5, 0.3]), ph)
        tr = 0.4
    else:  # KIND_BACK
        f0 = rng.uniform(380, 520)
        body = modal(t, f0 * np.array([1.0, 2.1, 3.3]),
                     np.array([0.012, 0.008, 0.005]), np.array([1.0, 0.5, 0.3]), ph)
        tr = 0.3
    # 0.4 ms attack ramp avoids an unnatural digital step
    att = np.minimum(t / 0.0004, 1.0)
    sig = body * att + tr * click(n, rng, hp_sos)
    amp = gain * (v / 100.0) ** 0.85
    return sig * amp


def room_ir(rng, seconds=0.45, rt60=0.32):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / rt60)
    ir = rng.standard_normal((2, n)) * env
    lp = butter(2, 6000, "low", fs=SR, output="sos")
    ir = sosfilt(lp, ir, axis=1)
    pre = int(0.006 * SR)                    # pre-delay
    ir[:, :pre] = 0
    for d, g in ((0.011, 0.5), (0.017, 0.35), (0.023, 0.3)):   # early reflections
        k = int(d * SR)
        ir[0, k] += g
        ir[1, k + 37] += g * 0.9
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


def main():
    sim_path, out_path = sys.argv[1], sys.argv[2]
    sim = np.load(sim_path)
    ev = sim["events"]
    rng = np.random.default_rng(99)
    total = int((B.DURATION + 1.0) * SR)
    dry = np.zeros((2, total))
    tables = peg_pitch_table()
    hp_sos = butter(2, 4000, "high", fs=SR, output="sos")

    kept = 0
    for t0, kind, v, x, z in ev:
        kind = int(kind)
        if v < KIND_PARAMS[kind][0] or t0 >= B.DURATION:
            continue
        g = grain(kind, v, x, z, rng, tables, hp_sos)
        i0 = int(t0 * SR)
        g = g[: total - i0]
        pan = np.clip(x / (B.HALF_W + 2), -1, 1) * 0.55
        ang = (pan + 1) * np.pi / 4
        dry[0, i0:i0 + len(g)] += g * np.cos(ang)
        dry[1, i0:i0 + len(g)] += g * np.sin(ang)
        kept += 1
    print(f"events used {kept} of {len(ev)}")

    # gentle tilt: tame harsh 5-9 kHz pile-up, keep air
    peak_sos = butter(2, [5200, 8800], "bandstop", fs=SR, output="sos")
    dry = 0.6 * dry + 0.4 * sosfilt(peak_sos, dry, axis=1)

    ir = room_ir(rng)
    wet = np.stack([fftconvolve(dry[c], ir[c])[:total] for c in range(2)])
    mix = dry + 0.22 * wet

    # very quiet room tone so silence does not sound digital
    pink = np.cumsum(rng.standard_normal((2, total)), axis=1)
    pink = sosfilt(butter(1, 40, "high", fs=SR, output="sos"), pink, axis=1)
    pink /= np.max(np.abs(pink)) + 1e-9
    mix += pink * 10 ** (-66 / 20)

    # normalise loudness on the busy section, then soft-clip peaks
    busy = mix[:, int(5 * SR):int(16 * SR)]
    rms = np.sqrt(np.mean(busy ** 2))
    mix *= 10 ** (-19 / 20) / rms
    mix = np.tanh(mix / 0.89) * 0.89

    # fade out the last second
    fade = int(1.2 * SR)
    end = int(B.DURATION * SR)
    mix[:, end - fade:end] *= np.linspace(1, 0, fade) ** 2
    mix = mix[:, :end]

    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", out_path, "peak", float(np.max(np.abs(mix))))


if __name__ == "__main__":
    main()
