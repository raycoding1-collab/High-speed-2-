"""
Checkpoint 7c: Fast low-res motion preview of the entire 75-second film (every shot S01..S10)
with burned-in physics telemetry HUD (speed km/h & m/s, camera speed, distance, focal length,
180-deg motion blur length, RECORD RUN tag) muxed with the EBU R128 synthesized soundtrack!
"""
import csv
import os
import subprocess
import sys
import time
import cv2
import numpy as np
import bpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import specs


def render_motion_preview(blend_path: str, physics_csv: str, audio_wav: str, out_mp4: str,
                          stride: int = 4, res=(640, 360), samples: int = 8):
    """
    Render every `stride`-th frame across the whole 1,800-frame film (e.g. stride=4 -> 450 frames @ 6 fps
    smoothly held to 24 fps, or stride=3 -> 600 frames) using fast Cycles CPU preview + OIDN,
    burn in the exact per-frame physics HUD, and mux with the 75 s stereo soundtrack!
    """
    phys = {}
    with open(physics_csv) as f:
        for row in csv.DictReader(f):
            phys[int(row["frame"])] = row

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    for sc in bpy.data.scenes:
        sc.render.resolution_x = res[0]
        sc.render.resolution_y = res[1]
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = False
        sc.cycles.max_bounces = 3
        sc.cycles.diffuse_bounces = 2
        sc.cycles.glossy_bounces = 2
        sc.cycles.use_denoising = True
        sc.render.use_persistent_data = True
        sc.render.image_settings.file_format = "PNG"

    prev_dir = os.path.join(REPO_ROOT, "build", "preview_frames")
    os.makedirs(prev_dir, exist_ok=True)
    t0 = time.time()
    all_frames = list(range(1, specs.TOTAL_FRAMES + 1, stride))
    prev_sc = None
    for idx, f in enumerate(all_frames):
        hud_p = os.path.join(prev_dir, f"hud_{idx:04d}.png")
        if os.path.exists(hud_p) and os.path.getsize(hud_p) > 2048:
            continue
        shot = specs.get_shot_for_frame(f)
        sc = bpy.data.scenes[f"Scene_{shot.loc_id}"]
        if prev_sc is not None and prev_sc != sc:
            prev_sc.render.use_persistent_data = False
        sc.render.use_persistent_data = True
        prev_sc = sc
        bpy.context.window.scene = sc
        sc.camera = bpy.data.objects[f"Cam_{shot.shot_id}"]
        sc.view_settings.exposure = shot.exposure_ev
        sc.frame_set(f)

        raw_p = os.path.join(prev_dir, f"raw_{idx:04d}.png")
        sc.render.filepath = raw_p
        bpy.ops.render.render(write_still=True)

        # Burn in crisp per-frame physics telemetry HUD!
        im = cv2.imread(raw_p)
        r = phys[f]
        rec_tag = " [RECORD RUN 603 km/h]" if r["is_record_run"] == "1" else ""
        line1 = f"{r['shot_id']} {shot.name}{rec_tag} | f{f:04d}/{specs.TOTAL_FRAMES} ({float(r['time_s']):05.2f}s)"
        line2 = (
            f"v={float(r['train_speed_kmh']):.0f} km/h ({float(r['train_speed_ms']):.1f} m/s) | "
            f"cam={float(r['cam_speed_ms']):.1f} m/s ({r['rig_type']}) | "
            f"d={float(r['cam_dist_m']):.0f}m | {float(r['focal_mm']):.0f}mm | blur={float(r['blur_length_m']):.2f}m"
        )
        overlay = im.copy()
        cv2.rectangle(overlay, (0, 0), (res[0], 42), (10, 12, 16), -1)
        im = cv2.addWeighted(overlay, 0.62, im, 0.38, 0)
        sc_f = res[0] / 640.0
        cv2.putText(im, line1, (8, int(16 * sc_f) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sc_f, (255, 240, 180), 1, cv2.LINE_AA)
        cv2.putText(im, line2, (8, int(33 * sc_f) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.37 * sc_f, (220, 240, 255), 1, cv2.LINE_AA)

        cv2.imwrite(hud_p, im)
        if (idx + 1) % 25 == 0:
            print(f"[preview] Rendered {idx + 1}/{len(all_frames)} preview frames (f{f}/{specs.TOTAL_FRAMES}) in {time.time() - t0:.1f}s")

    # Encode preview MP4 at exact 75.0 s duration synced to full EBU R128 soundtrack!
    eff_fps = specs.FPS / stride
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", f"{eff_fps:.4f}",
        "-i", os.path.join(prev_dir, "hud_%04d.png"),
        "-i", audio_wav,
        "-r", str(specs.FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", out_mp4
    ], check=True)
    print(f"[preview] Wrote {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.2f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default=os.path.join(REPO_ROOT, "build", "film.blend"))
    p.add_argument("--physics", default=os.path.join(REPO_ROOT, "docs", "physics_log.csv"))
    p.add_argument("--audio", default=os.path.join(REPO_ROOT, "build", "soundtrack.wav"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "review", "motion_preview_hud.mp4"))
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--samples", type=int, default=4)
    args = p.parse_args()
    render_motion_preview(args.blend, args.physics, args.audio, args.out,
                          stride=args.stride, res=(args.width, args.height), samples=args.samples)
