"""
Resilient, interleaved Blender Cycles frame renderer for GitHub Actions 20-job pipeline.

Features:
- Interleaved assignment: job `k` (0..N-1) renders frames `k+1, k+1+N, ...` for balanced load.
- Supports chunking (`--chunk-index`, `--chunk-size 10`) so workflows can upload artifacts
  every 10 frames (surviving vanished runners!).
- Strict resume check: skips any existing PNG ONLY IF it ends with the PNG `IEND` chunk
  AND decodes cleanly without truncation via PIL.
- Automatically selects the exact Blender Scene (`Scene_<loc_id>`), active Camera (`Cam_<shot_id>`),
  and per-shot DP exposure (`shot.exposure_ev`) for each frame.
- Writes per-frame timing JSONL so actual cloud s/frame and total runner-hours are measured accurately.
"""
import argparse
import json
import os
import sys
import time
from PIL import Image
import bpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import specs


def is_valid_complete_png(path: str, expected_size=None) -> bool:
    """Verify file exists, contains valid PNG IEND trailer, and decodes completely."""
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                return False
            f.seek(max(0, os.path.getsize(path) - 64))
            tail = f.read(64)
            if b"IEND" not in tail:
                return False
        with Image.open(path) as im:
            im.load()
            if expected_size and im.size != tuple(expected_size):
                return False
        return True
    except Exception:
        return False


def get_assigned_frames(job_index: int, total_jobs: int, frame_start: int, frame_end: int,
                        explicit_frames=None) -> list:
    if explicit_frames is not None:
        return sorted(set(explicit_frames))
    return [f for f in range(frame_start, frame_end + 1) if ((f - 1) % total_jobs) == job_index]


def render_frames(
    blend_path: str,
    out_dir: str,
    frames: list,
    res=None,
    samples: int = None,
    timing_log: str = None,
):
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Optimize CPU threads for GitHub Actions runner (uses all available vCPUs)
    for sc in bpy.data.scenes:
        sc.render.threads_mode = "AUTO"
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        if res is not None:
            sc.render.resolution_x = res[0]
            sc.render.resolution_y = res[1]
        if samples is not None:
            sc.cycles.samples = samples
            sc.cycles.adaptive_min_samples = max(8, samples // 4)

    expected_sz = tuple(res) if res else None
    rendered, skipped = 0, 0

    for f in frames:
        shot = specs.get_shot_for_frame(f)
        out_png = os.path.join(out_dir, f"frame_{f:04d}.png")
        if is_valid_complete_png(out_png, expected_size=expected_sz):
            skipped += 1
            print(f"[worker] SKIP valid frame {f:04d} ({shot.shot_id})")
            continue

        sc = bpy.data.scenes[f"Scene_{shot.loc_id}"]
        bpy.context.window.scene = sc
        sc.camera = bpy.data.objects[f"Cam_{shot.shot_id}"]
        sc.view_settings.exposure = shot.exposure_ev
        sc.frame_set(f)

        # Render to temp path then atomic rename so partial writes never pollute out_dir
        tmp_png = os.path.join(out_dir, f".tmp_frame_{f:04d}.png")
        sc.render.filepath = tmp_png

        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0

        if not is_valid_complete_png(tmp_png, expected_size=expected_sz):
            raise RuntimeError(f"Rendered frame {f:04d} failed PNG IEND/decode verification!")
        os.replace(tmp_png, out_png)
        rendered += 1
        print(f"[worker] DONE frame {f:04d} ({shot.shot_id} / {shot.loc_id}) in {dt:.2f}s")

        if timing_log:
            if os.path.dirname(timing_log):
                os.makedirs(os.path.dirname(timing_log), exist_ok=True)
            with open(timing_log, "a") as tf:
                tf.write(json.dumps({
                    "frame": f, "shot_id": shot.shot_id, "loc_id": shot.loc_id,
                    "sec": round(dt, 3), "samples": sc.cycles.samples,
                    "res": [sc.render.resolution_x, sc.render.resolution_y],
                }) + "\n")

    print(f"[worker] Batch complete: rendered={rendered}, skipped={skipped}, total={len(frames)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--job-index", type=int, default=0)
    p.add_argument("--total-jobs", type=int, default=20)
    p.add_argument("--frame-start", type=int, default=1)
    p.add_argument("--frame-end", type=int, default=specs.TOTAL_FRAMES)
    p.add_argument("--chunk-index", type=int, default=None, help="0-based chunk of size --chunk-size within this job")
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--frames", type=int, nargs="*", default=None, help="Explicit frame numbers (for benchmark or patch)")
    p.add_argument("--frames-file", type=str, default=None, help="JSON list of explicit frame numbers")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--timing-log", type=str, default=None)
    args = p.parse_args()

    explicit = args.frames
    if args.frames_file and os.path.exists(args.frames_file):
        with open(args.frames_file) as f:
            explicit = json.load(f)

    assigned = get_assigned_frames(args.job_index, args.total_jobs, args.frame_start, args.frame_end, explicit)
    if args.chunk_index is not None:
        c0 = args.chunk_index * args.chunk_size
        c1 = c0 + args.chunk_size
        assigned = assigned[c0:c1]

    if not assigned:
        print("[worker] No frames in this chunk; exiting cleanly.")
        return

    res = (args.width, args.height) if (args.width and args.height) else None
    render_frames(args.blend, args.out_dir, assigned, res=res, samples=args.samples, timing_log=args.timing_log)


if __name__ == "__main__":
    main()
