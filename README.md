# 500 km/h Through Japan — JR Central L0 Series SCMaglev CGI Short Film

A hyper-realistic 75-second CGI short film (**1,800 frames @ 24 fps, 1920×1080**, plus 4K YouTube master and <25 MB Instagram-DM encode) of Japan's **JR Central L0 Series (Improved 950-Series) SCMaglev** operating at **500 km/h (138.9 m/s)** service speed — plus a dedicated **603 km/h (167.5 m/s)** record-run sequence — across real elevation corridors of the Chūō Shinkansen / Yamanashi Maglev Line.

Built 100% programmatically in **Blender 4.5 (`bpy`, Cycles CPU)** with **real Copernicus GLO-30 30 m satellite DEM terrain**, **CC0 Poly Haven photogrammetry PBR textures, 3D conifers, boulders & golden-hour HDRI**, physically simulated train/camera kinematics, a physics-synthesized **EBU R128 (−23 LUFS)** stereo soundtrack, and a resumable **20-job GitHub Actions cloud render pipeline**.

## Repository Layout
- `src/specs.py` — Single source of truth for all metric train, guideway, curve/cant, DEM location, and 10-shot camera specs.
- `src/physics.py` — Exact 1,800-frame kinematics & camera rig solver (writes `docs/physics_log.csv`).
- `src/assets/` — Procedural L0-950 train (`train_l0.py`), double-track guideway/viaduct/buffer & noise hoods (`guideway.py`), Copernicus DEM & HDRI fetcher (`fetch_dems.py`), CC0 Poly Haven photogrammetry fetcher & triplanar PBR builder (`fetch_polyhaven_cc0.py`, `pbr_materials.py`), Japanese autumn vegetation (`vegetation.py`), and *kawara*-roof hamlet/shrine/utility poles (`buildings.py`).
- `src/scenes/` — Golden-hour HDRI+Sun lighting & AgX color pipeline (`lighting.py`), 6 real-DEM Japanese autumn locations (`environments.py`), and packed master scene assembler (`build_film.py`).
- `src/audio/synth_soundtrack.py` — Physically propagated stereo soundtrack (retarded-time Doppler, 1/r, air absorption, bow-wave pressure pulse, tunnel boom, autumn ambience) mastered to EBU R128.
- `src/verify/` — OpenCV `solvePnP` reference-photo camera solver (`solve_pnp_train.py`), automated Section 6 verification gates (`verify_gates.py`), and HUD motion preview (`render_preview.py`).
- `src/pipeline/` — Interleaved chunked Cycles worker with PNG `IEND` resume (`render_worker.py`) and safe multi-target FFmpeg encoder/verifier (`encode_deliverables.py`).
- `.github/workflows/` — Cloud benchmark (`benchmark.yml`), 20-job periodic-chunk final render (`render_film.yml`), and multi-batch / missing-frame patcher (`patch_missing.yml`).
- `deliveries/` — Final verified film masters committed directly to git (`film_1080p.mp4`, `film_instagram.mp4` <25 MB) & 3 hero `1280×720` thumbnails.
- `docs/` — Research & citations (`research_l0_scmaglev.md`), route & shot sheet (`route_and_shots.md`), per-frame physics log (`physics_log.csv`), EBU R128 loudness report (`loudness_report.json`), full 3-master verification report (`ffprobe_report.json`), verification checklist (`verification_checklist.md`), and YouTube metadata & CGI disclosure (`youtube_metadata.md`).
- `review/` — Compact JPG review stills and 75 s HUD motion preview (`motion_preview_hud.mp4`).

## Final Deliverables (`ffprobe`-Verified, 1,800 / 1,800 Frames)
| Deliverable | Resolution | Bitrate | File Size | Color / Audio | Location |
|---|---|---|---|---|---|
| **1080p Master** | `1920×1080` @ `24 fps` (`75.0 s`) | `9.13 Mbps` | `85.62 MB` | `yuv420p`, BT.709, AAC `48 kHz` stereo (`−23.00 LUFS`) | [`deliveries/film_1080p.mp4`](deliveries/film_1080p.mp4) |
| **Instagram-DM Copy (<25 MB)** | `1920×1080` @ `24 fps` (`75.0 s`) | `2.35 Mbps` | `22.00 MB` | `yuv420p`, BT.709, AAC `48 kHz` stereo (`−23.00 LUFS`) | [`deliveries/film_instagram.mp4`](deliveries/film_instagram.mp4) |
| **YouTube 4K Master** | `3840×2160` @ `24 fps` (`75.0 s`) | `51.09 Mbps` | `478.94 MB` | `yuv420p`, BT.709, AAC `320 kbps` `48 kHz` stereo | GitHub Actions `final-film-deliverables` artifact (too large for git) |
| **3 Hero Thumbnails** | `1280×720` JPG | — | `206–423 KB` | Frames `287` (`S02`), `464` (`S03`), `1204` (`S07`) | [`deliveries/`](deliveries/) |

### Cloud Render Summary (`1920×1080`, `48 spp` Cycles CPU + OIDN, 0 Failed Jobs)
- **Batch 1 (`frames 1–900`):** [GitHub Actions Run `#45084189424`](https://github.com/raycoding1-collab/High-speed-2-/actions/runs/45084189424) — `21/21` jobs succeeded (`0` failed), `6h 12m 37s`.
- **Batch 2 (`frames 901–1800`):** [GitHub Actions Run `#45120453166`](https://github.com/raycoding1-collab/High-speed-2-/actions/runs/45120453166) — `21/21` jobs succeeded (`0` failed), `7h 07m 55s`.
- **Verification:** `1,800 / 1,800` PNGs validated (`IEND` + PIL decode, `0` missing/corrupt), 10/10 encoded-video shot frames visually verified, audio mastered to `−23.00 LUFS` / `−4.50 dBTP` (EBU R128 pass).

## Licensing & Attribution
- All code and procedural assets in this repository are original / self-made.
- All third-party textures, HDRI (`kloppenheim_06_puresky`), and 3D tree/rock scans are **CC0 (Public Domain)** from [Poly Haven](https://polyhaven.com/).
- Elevation data is open **Copernicus DEM GLO-30 (30 m)** via AWS Open Data.
- No copyrighted or CC-BY-SA reference photographs are committed to this repository.

## Reproducing Locally
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/scenes/build_film.py --out build/film.blend --samples 48
python src/audio/synth_soundtrack.py
```
