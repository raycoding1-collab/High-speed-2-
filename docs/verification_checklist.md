# Pre-Render Self-Review & Section 6 Verification Checklist (Checkpoint 7e — Final Gold Master)

**Film:** *"500 km/h Through Japan"* — JR Central L0 Series SCMaglev (Improved 950-Series)  
**Spec:** 75.0 s (`1,800` frames @ `24 fps`), `1920×1080` master (+ `3840×2160` YouTube 4K + `<25 MB` Instagram-DM)  
**Decision:** **ALL QUANTITATIVE & VISUAL GATES VERIFIED — CLEARED FOR FULL 2-BATCH CLOUD RENDER**

---

## 1. Automated Quantitative Gates (Code-Measured)

| Gate | Status | Verification Method | Measured Evidence |
|---|---|---|---|
| **Train proportions & camera alignment** | **Verified** | OpenCV `solvePnP` against 2 independent real Yamanashi L0-950 photos (`src/verify/solve_pnp_train.py`) | `ref1_L0_950`: RMS `19.52 px` (`90.9 mm` eq.); `ref2_Yamanashi5`: RMS `11.30 px` (`74.1 mm` eq.) |
| **Levitation gap & wheel retraction** | **Verified** | Dedicated ruler closeup (`review/cp7a_train_levitation_ruler.jpg`) + `specs.py` assertion | Exact `10 cm` (`0.10 m`) clearance above trough floor; wheels retracted above `150 km/h` |
| **Adaptive cinema shutter & ultra-slow-mo ramps** | **Verified** | `physics.py` + `render_worker.py` (`motion_blur_shutter = frame_shutter_fraction(f)`) + `docs/physics_log.csv` | Real-time `1.0×` (`180°` shutter = `0.50` frames): exact `2.894 m` blur @ `500 km/h` & `3.490 m` @ `603 km/h`. Showstopper ultra-slow-mo plateaus (`72°` adaptive shutter = `0.20` frames): `S02` (`2.66 s @ 0.08×`, `9.3 cm` razor-sharp blur) & `S07` (`2.80 s @ 0.10×`, `11.6 cm` razor-sharp blur) |
| **Evaluated depsgraph & NaN audit** | **Verified** | Live Blender 4.5 depsgraph traversal across all 6 scenes (`src/verify/verify_gates.py`) | `loc1`: `9,001`, `loc2`: `9,813`, `loc3`: `7,387`, `loc4`: `5,713`, `loc5`: `5,311`, `loc6`: `7,609`; **`0` NaN/Inf**; Peak RAM **`4.41 GB`** (well within 16 GB GitHub runner RAM) |
| **100% packed CC0 assets** | **Verified** | `build_film.py` strict unpacked-image assertion + `docs/cc0_assets_provenance.json` | `95/95` images packed inside `build/film.blend` (`377.6 MB`); 100% CC0 Poly Haven (8 PBR sets + 5 3D photogrammetry libraries) + Copernicus GLO-30 DEM + procedural code; zero copyrighted/CC-BY-SA photos committed |
| **Shadow neutrality & exposure** | **Verified** | Automated shadow-region RGB balance & histogram audit (`docs/verification_report.json`) | `shadow_teal_excess < +0.085` on all 10 shots; `clip = 0.0%–0.07%`; mean luma `0.191–0.543` (**10/10 PASS**) |
| **75 s HUD motion preview** | **Verified** | Full 75 s `render_preview.py` (`review/motion_preview_hud.mp4`) + `ffprobe` | Exact `1,800` frames (`75.00 s` @ `24 fps`), H.264 + AAC stereo, `2.86 MB` |
| **EBU R128 theatrical physics soundtrack** | **Verified** | `src/audio/synth_soundtrack.py` + `docs/loudness_report.json` | `48 kHz` stereo, `75.0 s`, integrated **`−23.00 LUFS`**, True Peak **`−4.50 dBTP`** (`≤ −1.0 dBTP`); includes time-ramp-locked sub-bass pitch-down drops, stretched airflow/EDS coil/leaf textures & crisp sonic snap-backs |
| **Cloud benchmark (1080p / 48 spp)** | **Verified** | GitHub Actions `benchmark.yml` Run `#45061717463` (20 frames, 2/shot) | `1h22m25s`; **`119.49 runner-hours`** (`5.97 h` single-batch -> split safely into 2× 20-job batches) |

---

## 2. Whole-Film Visual & Cinema Craft Audits (Direct Frame Inspection Across All 10 Shots)

Every flaw below was caught by direct visual frame inspection across `S01..S10` and resolved before final cloud launch:

- **True golden-hour cinema contrast (`sun=12.0`, `sky=0.34` @ `302° / 11.5°`):** Fixed an override in `build_film.py` that had held contrast flat at `7.2 / 0.78`; unlocked blazing warm `3800 K` raking light, deep cool shadows, and 35 mm cinema Fog Glow glare across all 10 shots.
- **Zero-halo atmospheric perspective (`Env`-pass mist):** Replaced flat grey mist + binary `0.998` cutoff with Cycles' per-pixel `Environment` pass (`Env`) blend (`Mist * 0.42`) — eliminated 100% of sky-edge ridge & tree-tip halos in `S05` and `S06` while keeping seamless `42 km` aerial perspective across the Misaka mountains and 36 km Mt Fuji.
- **Exposed Yamanashi U-trough guideway:** Replaced solid `1.85 m` tan outer parapet with `0.42 m` concrete edge curb + dual open dark-steel handrails, cool-neutral weathered Japanese concrete grey, precast seams, rainwater streaking, beveled piers, and crisp figure-8 sidewall coils.
- **Real 3D CC0 autumn foliage & rice tufts:** Replaced all flat cutout-card autumn maples with CC0 Poly Haven 3D `tree_small_02_LOD1` deciduous trees, and replaced yellow sticks in `S07` with CC0 3D `grass_medium_01` harvested straw tufts over dark reflective paddy water.
- **Macro landscape & water realism (`S03`, `S04`, `S05`, `S06`, `S08`, `S10`):**
  - Added multi-scale forested evergreen/crimson canopy shading + craggy alpine rock-ridge bump & crisp snow couloirs (`S03`/`S05`), plus `valley_base_z = -260 m` and 2,600 midground ridge trees in `S05` (eliminated the horizontal tan elevation band).
  - Extended Kōfu mid-basin forest (`S03`) and dressed `S04` with a Japanese white galvanized W-beam roadside guardrail, `1,650` 3D meadow tufts & `230` autumn hedgerow shrubs.
  - Curved the Tenryū River gorge behind a forested mountain shoulder (`S06`, 100% eliminated the horizon V-notch) and upgraded river water to deep mineral jade-emerald with anisotropic rapids & foam streaks.
  - Replaced the straight linear exit-portal mountain wall in `S08` with a `tanh`-capped organic forested mountain dome carpeted with `7,400` 3D trees.
  - Widened pier spans to `38.0 m`, shifted `S10` off-axis (`78 mm`, `13°`, `+0.80 EV`), removed ghostly parapet masts, anchored all piers `-45 m` into valley slopes, and removed noisy `RiverMist` volume boxes.
- **2 Earned Cinema Showstopper Beats (`S02` & `S07`):**
  - **`S02`:** Grounded trackside jib push-in (`0.45 m` rail arc) + `35 → 52 mm` lens snap into `2.66 s @ 0.08×` (`72°` shutter, `9.3 cm` blur) showcasing the gleaming nose, `10 cm` levitation gap & figure-8 coils, paired with sub-bass drop, stretched EDS coil resonance, and punchy sonic snap-back.
  - **`S07`:** Fluid-head whip-pan + `44 → 62 mm` zoom push-in into `2.80 s @ 0.10×` (`72°` shutter, `11.6 cm` blur) dead-center over the vermilion Torii & cedar shrine with 180 slow-motion crimson maple leaves, paired with leaf-flutter & stretched *bonsho* bell audio + sonic snap-back.

---

## 3. Safe 2-Batch Overnight Cloud Render Plan (`1920×1080`, `48 spp` Cycles + OIDN)

1. **Batch 1 (`render_film.yml`):** `frame_start=1, frame_end=900, samples=48, run_encode=false` — 20 parallel jobs × 45 frames (~`2.5–2.8 h`, uploads every 10 frames).
2. **Batch 2 (`render_film.yml`):** `frame_start=901, frame_end=1800, samples=48, run_encode=false` — 20 parallel jobs × 45 frames (~`2.9–3.2 h`).
3. **Merge, Patch & Multi-Target Encode (`patch_missing.yml`):** Merges both runs safely (downloading `frames-*` chunks + single `soundtrack.wav` to keep runner disk lean), verifies all `1,800` PNGs (`IEND` + PIL decode), auto-patches any missing frame, encodes `film_1080p.mp4`, `film_instagram.mp4 (<25 MB)`, `film_youtube_4k.mp4`, and 3 hero thumbnails, verifies via `ffprobe`, and commits/uploads deliverables.

---

## 4. Honest Remaining Limitations
- High-altitude 10–36 km DEM peaks (`S03` Mt Fuji / Misaka, `S05` Minami Alps) originate from 30 m Copernicus GLO-30 elevation grids, so macro ridgeline silhouettes remain slightly smoother on frozen stills than photogrammetry foreground cliffs, even with procedural canopy/crag normal breakup.
- Background village farmhouses (`S03`, `S04`, `S07`) use 6 distinct procedural Japanese architectures with scanned 2K PBR maps rather than hero photogrammetry building scans.
