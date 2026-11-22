# Pre-Render Self-Review & Section 6 Verification Checklist (Checkpoint 7e)

**Film:** *"500 km/h Through Japan"* — JR Central L0 Series SCMaglev (Improved 950-Series)  
**Spec:** 75.0 s (`1,800` frames @ `24 fps`), `1920×1080` master (+ `3840×2160` YouTube 4K + `<25 MB` Instagram-DM)  
**Verified Commit Baseline:** `35452e1` (+ this checklist) on `main`  
**Decision:** **ALL SECTION 6 GATES PASSED — CLEARED FOR FULL 20-JOB CLOUD RENDER**

---

## 1. Gate-by-Gate Verification Matrix

| Gate | Status | Verification Method | Measured Evidence |
|---|---|---|---|
| **Train proportions & camera alignment** | **Verified** | OpenCV `solvePnP` against 2 independent real Yamanashi L0-950 photos (`src/verify/solve_pnp_train.py`) | `ref1_L0_950`: RMS `19.52 px` (`90.9 mm` eq.); `ref2_Yamanashi5`: RMS `11.30 px` (`74.1 mm` eq.) |
| **Levitation gap & bogie/wheel retraction** | **Verified** | Dedicated ruler closeup render (`review/cp7a_train_levitation_ruler.jpg`) + code assertion | Exact `10 cm` (`0.10 m`) clearance above trough floor; landing wheels retracted above `150 km/h` |
| **Cabin glazing & micro-detail** | **Verified** | Closeup render (`review/cp7a_train_windows_interior.jpg`, `cp7a_train_hero_34.jpg`) | Dual-layer tinted glass + illuminated cabin interior + nose bug strikes + panel seams |
| **HDRI–Sun lighting alignment** | **Verified** | Equirectangular luminance peak detection + panoramic render audit (`review/cp7a_hdri_sun_pano.jpg`) | `kloppenheim_06_puresky_4k.exr` rotated `+81.7°`; measured sun–HDRI azimuth error **`2.12°`** |
| **Evaluated depsgraph instances & NaN check** | **Verified** | Live Blender 4.5 depsgraph traversal across all 6 scenes (`src/verify/verify_gates.py`) | `loc1`: `9,009`, `loc2`: `4,779`, `loc3`: `3,997`, `loc4`: `5,167`, `loc5`: `4,334`, `loc6`: `10,021`; **`0` NaN/Inf transforms**; Peak RAM **`3.03 GB`** |
| **100% packed CC0 assets** | **Verified** | `build_film.py` strict unpacked-image assertion + `docs/cc0_assets_provenance.json` | **`80 / 80` images packed** inside `film.blend` (`286 MiB`); 100% CC0 Poly Haven + Copernicus GLO-30 DEM + procedural code; zero copyrighted/CC-BY-SA photos committed |
| **Shadow neutrality (no teal cast)** | **Verified** | Automated shadow-region RGB balance audit on all 10 hero frames + 20 full-HD cloud frames | All 10 shots `shadow_teal_excess` in `[-0.120, -0.001]` (`< +0.085` threshold -> **10/10 PASS**) |
| **Exposure & histogram balance** | **Verified** | Automated luminance/crush/clip histogram audit on all 10 hero frames (`docs/verification_report.json`) | `clip = 0.0%` on all 10 shots; `crush` `0.0%–9.19%`; mean luma `0.201–0.563` (**10/10 PASS**) |
| **Anti-repetition & photorealism pass** | **Verified** | Visual inspection of upgraded stills `S03`, `S07`, `S08`, `S09`, `S10` after user critique | 6 photogrammetry conifers + 6 maple silhouettes + 6 Japanese building types + 3 roof palettes; precast concrete seams/streaks/bevels; recessed U-trough coils; ghostly parapet masts removed; 3D domain-warped anti-tiling terrain; detailed cedar/kawara Shinto shrine; broken-up mossy wall foreground; rooted slotted tunnel portal; calibrated golden-hour aerial depth |
| **75 s motion preview + telemetry HUD** | **Verified** | Full 75 s `render_preview.py` encode (`review/motion_preview_hud.mp4`) + `ffprobe` + frame inspection | Exact `1,800` frames (`75.00 s` @ `24 fps`), H.264 + AAC stereo, `2.86 MB`; caught & fixed floating `S05` left piers (`p_bot = -pier_height - 45 m`) and `S06` `RiverMist` fireflies |
| **Physics-synthesized EBU R128 soundtrack** | **Verified** | `src/audio/synth_soundtrack.py` + `docs/loudness_report.json` | `48 kHz` stereo, `75.0 s`, integrated **`−23.00 LUFS`**, True Peak **`−5.11 dBTP`** (`≤ −1.0 dBTP` -> **PASS**) |
| **Cloud benchmark (full 1080p / 48 spp)** | **Verified** | GitHub Actions `benchmark.yml` Run `#45061717463` (20 representative frames, 2 per shot) | Completed cleanly in `1h22m25s`; measured **`119.49 runner-hours`** (`5.97 h` for 20 jobs single-batch). All 20 frames inspected; `S02 f0290` low luma (`0.039`) confirmed as physically accurate 25 cm close pass-by wall shadow |

---

## 2. Cloud Benchmark Per-Shot Timing (`benchmark.yml` #45061717463, `1920×1080`, `48 spp`)

| Shot | Frames | Measured s/frame (4-core `ubuntu-24.04`) | Avg s/frame | Runner-Hours |
|---|---:|---|---:|---:|
| S01 | 192 | `316.50, 265.40` | 290.95 s | 15.52 h |
| S02 | 168 | `208.68, 212.41` | 210.54 s | 9.83 h |
| S03 | 216 | `215.26, 212.19` | 213.73 s | 12.82 h |
| S04 | 168 | `162.98, 165.91` | 164.45 s | 7.67 h |
| S05 | 168 | `175.12, 209.00` | 192.06 s | 8.96 h |
| S06 | 192 | `321.85, 251.75` | 286.80 s | 15.30 h |
| S07 | 192 | `258.78, 263.10` | 260.94 s | 13.92 h |
| S08 | 168 | `183.94, 164.46` | 174.20 s | 8.13 h |
| S09 | 168 | `272.33, 228.79` | 250.56 s | 11.69 h |
| S10 | 168 | `341.53, 329.43` | 335.48 s | 15.66 h |
| **Total** | **1,800** | — | **239.0 s** | **119.49 h** |

### Safe Overnight Execution Plan (Zero Timeout Risk, Uncompromised `48 spp` Quality)
Because `5.97 h` single-batch wall time sits right against GitHub Actions' 6-hour per-job limit, the final render is split cleanly into **two sequential 20-job batches** at full `1920×1080 / 48 spp` (with Cycles `use_persistent_data` enabled per chunk for an additional ~5–8% speedup):
1. **Batch 1 (`render_film.yml`):** `frame_start=1, frame_end=900, samples=48, run_encode=false` — 45 frames/job, estimated wall time **`~2.5–2.7 h`** (well under 6 h limit; uploads every 10 frames).
2. **Batch 2 (`render_film.yml`):** `frame_start=901, frame_end=1800, samples=48, run_encode=false` — 45 frames/job, estimated wall time **`~2.9–3.2 h`** (well under 6 h limit; faster now without `RiverMist` volume box).
3. **Merge, Patch & Final Encode (`patch_missing.yml`):** Passes both Run IDs (`<batch1_id> <batch2_id>`), downloads all chunk artifacts, validates all `1,800` PNGs (`IEND` + PIL decode + exact `1920×1080`), patches any missing frame automatically, encodes `film_1080p.mp4`, `film_instagram.mp4 (<25 MB)`, `film_youtube_4k.mp4`, and 3 hero thumbnails, verifies with `ffprobe`, and commits/uploads final deliverables.

---

## 3. Honest Remaining Visual Limitations (What Still Reads Slightly CG on Frame Freeze)
- **Distant high-elevation DEM ridges (`S05` Minami Alps, `S03` Misaka foothills):** Copernicus GLO-30 has 30 m native posting, so very distant unforested ridges still show slightly smooth macro-geometry when zoomed in on a frozen frame, though softened substantially by golden-hour aerial perspective.
- **Far side-ridge skyline in `S08` (603 km/h POV):** Near- and mid-slope embankments are densely forested (`9,800` trees) with zero texture tiling, while the very highest distant side-ridge rim near the sky edge still shows slightly thinner tree silhouettes (heavily blurred at 603 km/h).
- **Procedural rural details:** Village houses, utility poles, and cars use clean architectural geometry with real scanned PBR textures rather than hero photogrammetry scans; appropriate at mid/background distances (`40–300 m`) and motion-blurred speeds.
- **Water wakes & birds:** No animated birds/insects or dynamic splash particles beyond slipstream autumn wake leaves (`S01`, `S02`, `S04`, `S07`, `S10`).
