#!/usr/bin/env bash
# Build the Galton board ASMR short end to end:
#   physics sim -> collision-driven audio -> Cycles frames -> 1080x1920 MP4
#
# usage: ./make_video.sh [WORK_DIR] [OUT.mp4]
# env:   PY      python with bpy, pybullet, numpy, scipy (default: python3)
#        SEED    simulation seed (default 5, picked for a symmetric bell curve)
#        HEIGHT  render height before upscaling to 1920 (default 1280)
#        SAMPLES Cycles samples per pixel, denoised (default 16)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HERE/build}"
OUT="${2:-$HERE/output/galton_asmr_short.mp4}"
PY="${PY:-python3}"
SEED="${SEED:-5}"
HEIGHT="${HEIGHT:-1280}"
SAMPLES="${SAMPLES:-16}"
FPS="$("$PY" -c "import sys; sys.path.insert(0, '$HERE'); import board; print(board.FPS)")"

mkdir -p "$WORK/frames" "$(dirname "$OUT")"

[ -f "$WORK/sim.npz" ] || "$PY" "$HERE/sim.py" "$WORK/sim.npz" "$SEED"
"$PY" "$HERE/audio.py" "$WORK/sim.npz" "$WORK/audio.wav"
# re-running resumes: frames that already exist are skipped
"$PY" "$HERE/scene.py" "$WORK/sim.npz" "$WORK/frames" --samples "$SAMPLES" --height "$HEIGHT"

ffmpeg -hide_banner -y -framerate "$FPS" -start_number 1 -i "$WORK/frames/f_%04d.png" \
  -i "$WORK/audio.wav" \
  -vf "scale=1080:1920:flags=lanczos,unsharp=5:5:0.35:5:5:0,format=yuv420p" \
  -c:v libx264 -preset slow -crf 15 -profile:v high -tune film \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart "$OUT"
echo "wrote $OUT"
