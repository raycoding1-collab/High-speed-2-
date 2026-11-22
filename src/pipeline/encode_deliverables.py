"""
Safe frame merger, strict completeness validator, and multi-target FFmpeg encoder
for "500 km/h Through Japan".

Strictly enforces Section 5 & Section 8:
1. Safe merge across downloaded chunk artifacts (NEVER uses `mv -n`, which errors on duplicates in coreutils 9.4!).
2. Refuses to encode (raises RuntimeError) if ANY frame 0001..1800 is missing, truncated,
   lacks PNG IEND, fails PIL decode, or has wrong dimensions!
3. Produces ALL deliverables:
   - `film_1080p.mp4`    : 1920x1080, 24 fps, H.264 High, yuv420p, BT.709 tagged, AAC 256k stereo
   - `film_instagram.mp4`: Strictly < 25 MB, two-pass x264 `-preset veryslow`, BT.709 tagged
   - `film_youtube_4k.mp4`: 3840x2160 high-bitrate Lanczos+unsharp YouTube master, BT.709 tagged
   - 3 Thumbnail candidates (1280x720) from hero frames
4. Verifies every output with `ffprobe` (frame count, duration, video/audio streams, BT.709, byte size)
   and writes `docs/ffprobe_report.json`.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import specs
from pipeline.render_worker import is_valid_complete_png


def safe_merge_frames(src_dirs: list, merged_dir: str, expected_size=(1920, 1080)) -> dict:
    """
    Safely merge PNG frames from any number of chunk/job directories into `merged_dir`.
    Handles duplicate frames gracefully without failing (unlike `mv -n` in coreutils 9.4!).
    """
    os.makedirs(merged_dir, exist_ok=True)
    copied, skipped_dup, rejected_corrupt = 0, 0, 0
    for sd in src_dirs:
        for path in sorted(glob.glob(os.path.join(sd, "**", "frame_*.png"), recursive=True)):
            fname = os.path.basename(path)
            dst = os.path.join(merged_dir, fname)
            if os.path.exists(dst) and is_valid_complete_png(dst, expected_size):
                skipped_dup += 1
                continue
            if not is_valid_complete_png(path, expected_size):
                rejected_corrupt += 1
                continue
            shutil.copy2(path, dst)
            copied += 1
    print(f"[merge] copied={copied}, skipped_dup={skipped_dup}, rejected_corrupt={rejected_corrupt}")
    return {"copied": copied, "skipped_dup": skipped_dup, "rejected_corrupt": rejected_corrupt}


def validate_all_frames(merged_dir: str, total_frames: int = specs.TOTAL_FRAMES,
                        expected_size=(1920, 1080)) -> list:
    """Return list of missing or invalid frame numbers (1..total_frames)."""
    missing = []
    for f in range(1, total_frames + 1):
        p = os.path.join(merged_dir, f"frame_{f:04d}.png")
        if not is_valid_complete_png(p, expected_size):
            missing.append(f)
    return missing


def run_ffprobe(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-count_frames", path
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {
        "file": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "duration_s": round(float(data["format"]["duration"]), 4),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "pix_fmt": v.get("pix_fmt"),
        "color_space": v.get("color_space"),
        "color_primaries": v.get("color_primaries"),
        "color_transfer": v.get("color_transfer"),
        "nb_read_frames": int(v.get("nb_read_frames", v.get("nb_frames", 0))),
        "has_audio": a is not None,
        "audio_codec": a.get("codec_name") if a else None,
        "audio_sr": int(a.get("sample_rate", 0)) if a else 0,
        "audio_channels": int(a.get("channels", 0)) if a else 0,
    }


def _run(cmd: list):
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def encode_all(merged_dir: str, audio_wav: str, out_dir: str, report_path: str,
               total_frames: int = specs.TOTAL_FRAMES, expected_size=(1920, 1080)):
    os.makedirs(out_dir, exist_ok=True)
    missing = validate_all_frames(merged_dir, total_frames, expected_size)
    if missing:
        miss_file = os.path.join(out_dir, "missing_frames.json")
        with open(miss_file, "w") as f:
            json.dump(missing, f)
        raise RuntimeError(f"REFUSING TO ENCODE: {len(missing)} missing/corrupt frames (first 20: {missing[:20]})! Saved {miss_file}")

    assert os.path.exists(audio_wav), f"Missing audio track: {audio_wav}"
    in_pattern = os.path.join(merged_dir, "frame_%04d.png")
    bt709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

    # 1. Master 1080p MP4 (H.264 High, yuv420p, BT.709, AAC 256k)
    mp4_1080 = os.path.join(out_dir, "film_1080p.mp4")
    _run([
        "ffmpeg", "-y", "-framerate", str(specs.FPS), "-i", in_pattern, "-i", audio_wav,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        *bt709, "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-shortest",
        mp4_1080,
    ])

    # 2. Instagram-DM copy STRICTLY < 25 MB (two-pass, veryslow preset!)
    # Target 22.0 MB total over 75 s -> video ~2,218 kbps + audio 128 kbps
    mp4_ig = os.path.join(out_dir, "film_instagram.mp4")
    dur_s = total_frames / specs.FPS
    target_v_kbps = int((22.0 * 8 * 1000) / dur_s) - 128
    passlog = os.path.join(out_dir, "x264_ig_2pass")
    _run([
        "ffmpeg", "-y", "-framerate", str(specs.FPS), "-i", in_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709",
        "-c:v", "libx264", "-preset", "veryslow", "-b:v", f"{target_v_kbps}k",
        "-maxrate", f"{int(target_v_kbps * 1.4)}k", "-bufsize", f"{int(target_v_kbps * 2.0)}k",
        "-pix_fmt", "yuv420p", *bt709, "-pass", "1", "-passlogfile", passlog, "-an", "-f", "mp4", "/dev/null"
    ])
    _run([
        "ffmpeg", "-y", "-framerate", str(specs.FPS), "-i", in_pattern, "-i", audio_wav,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709",
        "-c:v", "libx264", "-preset", "veryslow", "-b:v", f"{target_v_kbps}k",
        "-maxrate", f"{int(target_v_kbps * 1.4)}k", "-bufsize", f"{int(target_v_kbps * 2.0)}k",
        "-pix_fmt", "yuv420p", *bt709, "-pass", "2", "-passlogfile", passlog, "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-shortest",
        mp4_ig,
    ])
    for pf in glob.glob(passlog + "*"):
        os.remove(pf)
    assert os.path.getsize(mp4_ig) < 25_000_000, f"Instagram MP4 exceeded 25 MB: {os.path.getsize(mp4_ig)}"

    # 3. YouTube 4K (3840x2160) high-bitrate upscale master (Lanczos + crisp micro-sharpen, BT.709)
    mp4_4k = os.path.join(out_dir, "film_youtube_4k.mp4")
    _run([
        "ffmpeg", "-y", "-framerate", str(specs.FPS), "-i", in_pattern, "-i", audio_wav,
        "-vf", "scale=3840:2160:flags=lanczos:out_color_matrix=bt709,unsharp=5:5:0.45:5:5:0.0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
        *bt709, "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2", "-shortest",
        mp4_4k,
    ])

    # 4. 3 Thumbnail Candidates (1280x720) from hero moments (Mt Fuji S03, Tunnel Portal S01, Tenryu Gorge S06)
    thumbs = [
        ("thumb_01_kofu_fuji.jpg", 464),
        ("thumb_02_portal_emergence.jpg", 54),
        ("thumb_03_tenryu_gorge.jpg", 1005),
    ]
    thumb_paths = []
    for tname, fnum in thumbs:
        src_p = os.path.join(merged_dir, f"frame_{fnum:04d}.png")
        dst_p = os.path.join(out_dir, tname)
        Image.open(src_p).convert("RGB").resize((1280, 720), Image.LANCZOS).save(dst_p, quality=94)
        thumb_paths.append(dst_p)

    # 5. FFprobe verification of all 3 deliverables
    probes = {
        "film_1080p": run_ffprobe(mp4_1080),
        "film_instagram": run_ffprobe(mp4_ig),
        "film_youtube_4k": run_ffprobe(mp4_4k),
        "thumbnails": [os.path.basename(p) for p in thumb_paths],
    }
    for key in ("film_1080p", "film_instagram", "film_youtube_4k"):
        pr = probes[key]
        assert pr["nb_read_frames"] == total_frames, f"{key} frame mismatch: {pr['nb_read_frames']} != {total_frames}"
        assert pr["has_audio"], f"{key} missing audio stream!"
        assert pr["pix_fmt"] == "yuv420p", f"{key} pix_fmt != yuv420p"
        assert pr["color_space"] == "bt709", f"{key} not tagged BT.709"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(probes, f, indent=2)
    print(json.dumps(probes, indent=2))
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dirs", nargs="+", required=True)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", default=os.path.join(REPO_ROOT, "docs", "ffprobe_report.json"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    safe_merge_frames(args.src_dirs, args.merged_dir, expected_size=(args.width, args.height))
    missing = validate_all_frames(args.merged_dir, specs.TOTAL_FRAMES, expected_size=(args.width, args.height))
    if args.check_only:
        out_m = os.path.join(args.out_dir, "missing_frames.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_m, "w") as f:
            json.dump(missing, f)
        print(f"[check] Missing frames: {len(missing)}")
        return
    encode_all(args.merged_dir, args.audio, args.out_dir, args.report, expected_size=(args.width, args.height))


if __name__ == "__main__":
    main()
