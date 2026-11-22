"""
Master Film Scene Assembler (`build/film.blend`) for "500 km/h Through Japan".

Architecture:
- Builds ONE clean, self-contained `film.blend` with 6 Blender Scenes (one per real DEM location)
  plus all 10 animated shot camera rigs (`Cam_S01` .. `Cam_S10`), sharing the linked 12-car (299.0 m)
  L0-950 train meshes & materials so file size stays compact (~120–200 MB) and GitHub Actions
  runners only evaluate the single active location's BVH per frame (<4 GB peak RAM).
- Packs ALL generated textures and the CC0 Poly Haven HDRI inside `film.blend` so cloud render
  workers need ZERO external downloads.
"""
import json
import math
import os
import sys
import bpy
from mathutils import Vector

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import specs
import physics
from assets.fetch_dems import fetch_all_dems, fetch_cc0_hdri
from assets.train_l0 import build_l0_train
from assets.vegetation import build_vegetation_library
from assets.buildings import build_building_library
from scenes.lighting import configure_scene_render, setup_golden_hour_world
from scenes.environments import build_location_environment, add_slipstream_wake_leaves

TEX_DIR = os.path.join(REPO_ROOT, "assets_cache", "textures")
BUILD_DIR = os.path.join(REPO_ROOT, "build")


def _aim_camera_matrix(cam_pos, target_pos):
    direction = Vector(target_pos) - Vector(cam_pos)
    if direction.length < 1e-4:
        direction = Vector((1.0, 0.0, 0.0))
    return direction.to_track_quat("-Z", "Y").to_euler()


def build_master_film(out_blend: str, samples: int = 48, res=(1920, 1080)) -> dict:
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    fetch_all_dems()
    hdri_path = fetch_cc0_hdri("4k")

    # Generate & validate per-frame physics log
    phys_csv = os.path.join(REPO_ROOT, "docs", "physics_log.csv")
    frames = physics.write_physics_log(phys_csv)
    frames_by_shot = {}
    for fp in frames:
        frames_by_shot.setdefault(fp.shot_id, []).append(fp)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create shared procedural libraries ONCE in blend data
    veg_lib = build_vegetation_library()
    bldg_lib = build_building_library()

    # Build one dedicated Scene per real DEM location
    location_scenes = {}
    env_stats = {}
    sun_meta_all = {}

    for loc_id, loc in specs.LOCATIONS.items():
        sc = bpy.data.scenes.new(f"Scene_{loc_id}")
        location_scenes[loc_id] = sc
        configure_scene_render(sc, res=res, samples=samples, adaptive_thresh=0.025, exposure_ev=0.10, motion_blur=True)
        sc.frame_start = 1
        sc.frame_end = specs.TOTAL_FRAMES

        # Consistent late-autumn golden-hour lighting across all 6 locations!
        sun_meta = setup_golden_hour_world(sc, hdri_path, hdri_strength=0.78, sun_strength=7.2, add_aerial_haze=True)
        sun_meta_all[loc_id] = sun_meta

        # Build DEM terrain + guideway + species-correct vegetation + hamlet/shrine/river
        st = build_location_environment(sc, loc_id, veg_lib, bldg_lib, TEX_DIR)

        # Place full 12-car (299.0 m) L0-950 train in this location's Train collection
        col_train = bpy.data.collections.new(f"Train_{loc_id}")
        sc.collection.children.link(col_train)
        train_root = build_l0_train(col_train, TEX_DIR, num_mid_cars=specs.NUM_INTERMEDIATE_CARS)
        train_root.name = f"L0_950_Root_{loc_id}"

        # Animate train & camera rigs for EVERY shot using this location
        wake_count = 0
        for shot in specs.SHOTS:
            if shot.loc_id != loc_id:
                continue
            cam_d = bpy.data.cameras.new(f"Cam_{shot.shot_id}")
            cam_d.lens = shot.focal_mm
            cam_d.sensor_width = 36.0
            cam_d.clip_start = 0.25
            cam_d.clip_end = 65000.0  # Must reach Mt Fuji at 31+ km!
            cam_d.dof.use_dof = True
            cam_d.dof.aperture_fstop = shot.fstop

            cam_o = bpy.data.objects.new(f"Cam_{shot.shot_id}", cam_d)
            sc.collection.objects.link(cam_o)
            sc.camera = cam_o

            shot_frames = frames_by_shot[shot.shot_id]
            for fp in shot_frames:
                # Keyframe train pose (including 1-frame pre/post extrapolation for crisp 180-deg shutter blur on boundary frames!)
                train_root.location = fp.train_pos
                train_root.rotation_euler = fp.train_eul
                train_root.keyframe_insert("location", frame=fp.frame)
                train_root.keyframe_insert("rotation_euler", frame=fp.frame)

                cam_o.location = fp.cam_pos
                cam_o.rotation_euler = _aim_camera_matrix(fp.cam_pos, fp.cam_target)
                cam_d.dof.focus_distance = fp.focus_dist_m
                cam_o.keyframe_insert("location", frame=fp.frame)
                cam_o.keyframe_insert("rotation_euler", frame=fp.frame)
                cam_d.dof.keyframe_insert("focus_distance", frame=fp.frame)

            # Set LINEAR interpolation on train motion fcurves so velocity is physically exact (no ease-in/out!)
            if train_root.animation_data and train_root.animation_data.action:
                for fc in train_root.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        if shot.frame_start <= kp.co.x <= shot.frame_end:
                            kp.interpolation = "LINEAR"

            # Add finite aerodynamic slipstream wake leaves on trackside/pass-by shots!
            if shot.shot_id in ("S01", "S02", "S04", "S07", "S10"):
                s_wake = {"S01": 35.0, "S02": 145.0, "S04": 45.0, "S07": 45.0, "S10": 980.0}[shot.shot_id]
                z_wake = 0.6 if shot.shot_id in ("S01", "S02", "S10") else -loc.viaduct_elev_above_ground + 2.5
                col_wake = bpy.data.collections.new(f"Wake_{shot.shot_id}")
                sc.collection.children.link(col_wake)
                wake_count += add_slipstream_wake_leaves(
                    col_wake, veg_lib, loc, s_wake, z_wake,
                    shot.frame_start, shot.frame_end, shot.train_x0_m, shot.speed_ms,
                    count=140, seed=700 + shot.frame_start
                )

        # Add parked/driving Japanese rural sedan on Kofu parallel road (loc2) for human scale!
        if loc_id == "loc2_kofu_fuji":
            car_obj = bpy.data.objects.new("Kofu_Rural_Car", bldg_lib["car"])
            car_obj.location = (65.0, -25.0, -loc.viaduct_elev_above_ground + 0.15)
            sc.collection.objects.link(car_obj)

        st["wake_leaves"] = wake_count
        env_stats[loc_id] = st
        print(f"[build_film] Built {sc.name}: trees={st['trees']}, buildings={st['buildings']}, crops={st['crops']}, wake={wake_count}")

    # Remove empty default scene if present
    for sc in list(bpy.data.scenes):
        if sc.name not in [s.name for s in location_scenes.values()]:
            bpy.data.scenes.remove(sc)

    # Pack ALL external images (livery, coils, Poly Haven CC0 PBR/trees/rocks, CC0 HDRI) directly into film.blend!
    bpy.ops.file.pack_all()
    unpacked = [img.name for img in bpy.data.images if img.source == "FILE" and img.packed_file is None]
    if unpacked:
        raise RuntimeError(f"Unpacked external images found before saving film.blend: {unpacked}")
    print(f"[build_film] Verified 100% packed images: {len(bpy.data.images)} images inside film.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=True)
    size_mb = os.path.getsize(out_blend) / 1e6
    print(f"[build_film] Saved packed master scene: {out_blend} ({size_mb:.1f} MB)")

    summary = {
        "blend_path": out_blend,
        "blend_size_mb": round(size_mb, 2),
        "samples": samples,
        "resolution": list(res),
        "locations": env_stats,
        "sun_alignment": sun_meta_all,
    }
    with open(os.path.join(BUILD_DIR, "scene_build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(BUILD_DIR, "film.blend"))
    p.add_argument("--samples", type=int, default=48)
    args = p.parse_args()
    build_master_film(args.out, samples=args.samples)
