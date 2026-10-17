# Galton Board ASMR — YouTube Short

A photoreal 3D render of ~200 rainbow marbles pouring through a walnut and
brass Galton board, settling into a bell curve. Vertical 1080×1920, 30 fps,
23 s, with a soundtrack in which every click is a real collision from the
simulation.

Output: `output/galton_asmr_short.mp4`

## How it is made

| Step | File | What it does |
|------|------|--------------|
| Geometry | `board.py` | Shared dimensions (cm) so colliders and visible meshes match exactly |
| Physics | `sim.py` | pybullet rigid bodies at 1 kHz; logs every contact onset with its approach speed |
| Sound | `audio.py` | Modal synthesis per collision: brass-pin "tink", marble click, wood "tok"; stereo pan by position, small-room reverb, mastered to −14 LUFS (YouTube's target) |
| Picture | `scene.py` | Blender 5.2 Cycles: lacquered walnut, smoked-oak back panel, domed brass pins, thin glass front, glossy marbles, studio area lights, depth of field and motion blur, camera pulls back from a macro close-up to the full board |
| Assembly | `make_video.sh` | Runs everything and encodes H.264 + AAC 320k |

Ball colours follow drop order, so the bins fill in rainbow layers. The seed
(5) was chosen from several runs for the most symmetric bell curve.

## Rebuild

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
PY=venv/bin/python ./make_video.sh            # ~4 h on 4 CPU cores
```

`HEIGHT=1920 SAMPLES=32` renders natively at 1080p (roughly 2.5× slower).
Rendering resumes where it stopped if interrupted.
