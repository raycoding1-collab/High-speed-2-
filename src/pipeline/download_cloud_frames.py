"""
Resilient, retry-hardened GitHub Actions frame-chunk downloader & validator.

Solves Azure Blob Storage transient `HTTP 503 Egress is over the account limit` errors
that abort bulk `gh run download` calls when downloading 200+ chunk artifacts concurrently:
- Paginates all artifacts via `gh api --paginate`
- Filters to real PNG chunk artifacts (`frames-*`, skipping tiny timing-only chunks)
- Downloads with bounded concurrency (3 workers) + exponential backoff retries (up to 8x)
- Validates every PNG (header + IEND + PIL decode + exact WxH) into `out_dir`
"""
import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from pipeline.encode_deliverables import is_valid_complete_png, validate_all_frames
import specs


def list_frame_artifacts(repo: str, run_id: str) -> list:
    cmd = ["gh", "api", "--paginate", f"repos/{repo}/actions/runs/{run_id}/artifacts"]
    out = subprocess.check_output(cmd, text=True)
    # `gh api --paginate` concatenates JSON objects; parse cleanly
    dec = json.JSONDecoder()
    pos = 0
    items = []
    while pos < len(out):
        while pos < len(out) and out[pos].isspace():
            pos += 1
        if pos >= len(out):
            break
        obj, pos = dec.raw_decode(out, pos)
        items.extend(obj.get("artifacts", []))
    # Keep only non-expired `frames-*` artifacts with real PNG payload (> 50 KB)
    return [
        a for a in items
        if not a.get("expired", False)
        and a["name"].startswith("frames-")
        and a.get("size_in_bytes", 0) > 50_000
    ]


def download_one_artifact(repo: str, run_id: str, art_name: str, out_dir: str,
                          expected_size=(1920, 1080), max_retries: int = 8) -> tuple:
    last_err = ""
    for attempt in range(1, max_retries + 1):
        tmp = tempfile.mkdtemp(prefix=f"dl_{run_id}_{art_name}_")
        try:
            res = subprocess.run(
                ["gh", "run", "download", str(run_id), "-R", repo, "-n", art_name, "-D", tmp],
                capture_output=True, text=True, timeout=180
            )
            if res.returncode != 0:
                last_err = (res.stderr or res.stdout).strip()
                shutil.rmtree(tmp, ignore_errors=True)
                wait_s = min(35, 2.5 * (1.6 ** (attempt - 1)))
                time.sleep(wait_s)
                continue

            copied = 0
            for png in glob.glob(os.path.join(tmp, "**", "frame_*.png"), recursive=True):
                fname = os.path.basename(png)
                dst = os.path.join(out_dir, fname)
                if os.path.exists(dst) and is_valid_complete_png(dst, expected_size):
                    continue
                if is_valid_complete_png(png, expected_size):
                    tmp_dst = os.path.join(out_dir, f".tmp_{fname}")
                    shutil.copy2(png, tmp_dst)
                    os.replace(tmp_dst, dst)
                    copied += 1
            shutil.rmtree(tmp, ignore_errors=True)
            return True, art_name, copied, ""
        except Exception as e:
            last_err = str(e)
            shutil.rmtree(tmp, ignore_errors=True)
            time.sleep(min(30, 3 * attempt))
    return False, art_name, 0, last_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="raycoding1-collab/High-speed-2-")
    p.add_argument("--run-ids", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    expected_sz = (args.width, args.height)

    tasks = []
    for rid in args.run_ids:
        arts = list_frame_artifacts(args.repo, rid)
        print(f"[dl] Run {rid}: found {len(arts)} PNG chunk artifacts", flush=True)
        for a in arts:
            tasks.append((rid, a["name"]))

    print(f"[dl] Downloading {len(tasks)} total chunks into {args.out_dir} (workers={args.workers})...", flush=True)
    done, failed = 0, []
    total_copied = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(download_one_artifact, args.repo, rid, name, args.out_dir, expected_sz)
            for (rid, name) in tasks
        ]
        for fut in concurrent.futures.as_completed(futs):
            ok, name, copied, err = fut.result()
            done += 1
            total_copied += copied
            if not ok:
                failed.append((name, err))
                print(f"[dl] FAIL {name}: {err[:160]}", flush=True)
            if done % 20 == 0 or done == len(tasks):
                have = len(glob.glob(os.path.join(args.out_dir, "frame_*.png")))
                print(f"[dl] Progress {done}/{len(tasks)} chunks ({time.time()-t0:.1f}s) | valid_frames={have} | failed={len(failed)}", flush=True)

    missing = validate_all_frames(args.out_dir, specs.TOTAL_FRAMES, expected_sz)
    print(f"[dl] COMPLETE: total_valid={specs.TOTAL_FRAMES - len(missing)}/{specs.TOTAL_FRAMES}, missing={len(missing)}, failed_chunks={len(failed)}", flush=True)
    if missing:
        print(f"[dl] First 20 missing: {missing[:20]}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
