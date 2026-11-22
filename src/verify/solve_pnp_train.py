"""
Checkpoint 7a Verification Script:
1. Builds the L0-950 train & U-guideway to exact metric specs (`src/specs.py`).
2. Solves camera extrinsics via OpenCV `cv2.solvePnP` against TWO real reference photos
   (stored in scratch, never committed to repo):
     - Ref A: L0-950.jpg (Improved L0-950 on Yamanashi viaduct, Sony EXIF 56.39 mm)
     - Ref B: Yamanashi_Maglev_Test_Track5.jpg (L0 on Yamanashi guideway)
   Renders the CGI L0-950 from each solved camera, overlays silhouette & wireframe edges
   onto the reference photo (saved to scratch/artifacts) and reports exact pixel reprojection
   residuals (RMS px).
3. Renders high-quality Checkpoint 7a train inspection stills (3/4 nose hero, low levitation-gap
   side view with 1 m scale ruler, interior window close-up) + verifies HDRI sun alignment panorama.
   Commits compact JPG review copies to `review/cp7a_*.jpg`.
"""
import json
import math
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
import bpy
from mathutils import Matrix, Vector

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import specs
from assets.train_l0 import build_l0_train, l0_nose_envelope
from assets.guideway import build_guideway_and_viaduct
from assets.fetch_dems import fetch_cc0_hdri
from scenes.lighting import configure_scene_render, setup_golden_hour_world

SCRATCH_REFS = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/90dafb5a-d9f7-4ccd-b9af-82f620e5f2e6/scratch/refs"
ARTIFACT_DIR = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/90dafb5a-d9f7-4ccd-b9af-82f620e5f2e6"
REVIEW_DIR = os.path.join(REPO_ROOT, "review")
TEX_DIR = os.path.join(REPO_ROOT, "assets_cache", "textures")


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_scale_ruler(collection, x_pos=-10.0, y_pos=-2.35, z_base=0.0):
    """Place a 4.0 m alternating black/white metric calibration ruler (1.0 m blocks) for Gate 6 scale verification."""
    import bmesh
    mat_w = bpy.data.materials.new("Ruler_W")
    mat_w.use_nodes = True
    mat_w.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    mat_k = bpy.data.materials.new("Ruler_K")
    mat_k.use_nodes = True
    mat_k.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.04, 0.04, 0.04, 1.0)
    mat_r = bpy.data.materials.new("Ruler_Red10cm")
    mat_r.use_nodes = True
    mat_r.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.85, 0.08, 0.05, 1.0)

    bm = bmesh.new()
    # 10 cm red marker block at bottom (showing exact 10 cm levitation gap!)
    ret = bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x_pos, y_pos, z_base + 0.05)) @ Matrix.Diagonal((0.18, 0.12, 0.10, 1.0)))
    for v in ret["verts"]:
        for f in v.link_faces:
            f.material_index = 2
    # 1.0 m vertical blocks up to 4.0 m
    for k in range(4):
        z0 = z_base + k * 1.0 + (0.10 if k == 0 else 0.0)
        h = 0.90 if k == 0 else 1.0
        ret = bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x_pos, y_pos, z0 + h * 0.5)) @ Matrix.Diagonal((0.14, 0.10, h, 1.0)))
        mi = k % 2
        for v in ret["verts"]:
            for f in v.link_faces:
                f.material_index = mi
    mesh = bpy.data.meshes.new("MetricRuler_4m")
    bm.to_mesh(mesh)
    bm.free()
    for _m in [mat_w, mat_k, mat_r]:
        mesh.materials.append(_m)
    obj = bpy.data.objects.new("MetricRuler_4m", mesh)
    collection.objects.link(obj)
    return obj


def solve_pnp_for_photo(img_path: str, pts_3d: np.ndarray, pts_2d: np.ndarray,
                        fov_x_deg_guess: float) -> dict:
    """
    Solve camera pose (rvec, tvec) and refine focal length via OpenCV solvePnP + 1D focal search
    to minimize RMS reprojection residual (px).
    """
    im = cv2.imread(img_path)
    H, W = im.shape[:2]
    cx, cy = W * 0.5, H * 0.5

    best = None
    # Sweep focal length around EXIF/initial guess to jointly optimize intrinsics + PnP pose
    f_guess = (W * 0.5) / math.tan(math.radians(fov_x_deg_guess * 0.5))
    for scale in np.linspace(0.65, 1.45, 33):
        fx = f_guess * scale
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - pts_2d, axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        if best is None or rms < best["rms_px"]:
            best = {
                "W": W, "H": H, "fx": float(fx), "K": K,
                "rvec": rvec, "tvec": tvec, "proj": proj,
                "rms_px": rms, "max_px": float(err.max()),
                "per_point_px": [round(float(e), 2) for e in err],
            }
    return best


def set_blender_cam_from_pnp(cam_obj, pnp: dict):
    """Convert OpenCV (rvec, tvec, K) directly into exact Blender camera world matrix & focal length."""
    R_cv, _ = cv2.Rodrigues(pnp["rvec"])
    t_cv = pnp["tvec"].reshape(3)
    # OpenCV camera coords: +X right, +Y down, +Z forward
    # Blender camera coords: +X right, +Y up, -Z forward -> flip Y and Z
    flip = np.diag([1.0, -1.0, -1.0])
    R_bl = flip @ R_cv
    t_bl = flip @ t_cv
    # World-to-camera -> Camera-to-world
    R_c2w = R_bl.T
    cam_pos = -R_c2w @ t_bl
    mat = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            mat[r][c] = float(R_c2w[r, c])
        mat[r][3] = float(cam_pos[r])
    cam_obj.matrix_world = mat
    cam_obj.data.sensor_fit = "HORIZONTAL"
    cam_obj.data.sensor_width = 36.0
    cam_obj.data.lens = 36.0 * (pnp["fx"] / pnp["W"])
    cam_obj.data.clip_start = 0.2
    cam_obj.data.clip_end = 5000.0


def verify_pnp_and_render_checkpoint():
    os.makedirs(REVIEW_DIR, exist_ok=True)
    hdri_path = fetch_cc0_hdri("4k")

    _reset_scene()
    scene = bpy.context.scene
    configure_scene_render(scene, res=(1280, 720), samples=36, adaptive_thresh=0.03, exposure_ev=0.15, motion_blur=False)
    sun_meta = setup_golden_hour_world(scene, hdri_path, hdri_strength=0.9, sun_strength=4.5, add_aerial_haze=False)

    # Build straight guideway & L0-950 train (7 cars for matching Ref A/B, or 12 cars — nose/lead cars identical!)
    loc_test = specs.LOCATIONS["loc6_record_straight"]
    build_guideway_and_viaduct(scene.collection, loc_test, TEX_DIR, s_start=-400.0, s_end=150.0, pier_height=14.0)
    train_root = build_l0_train(scene.collection, TEX_DIR, num_mid_cars=5)  # 7-car for matching real test photos!
    train_root.location = (0.0, 0.0, 0.0)

    # =========================================================================
    # A. HDRI + SUN PANORAMA VERIFICATION (Section 2: prevent 180-deg error!)
    # =========================================================================
    cam_data = bpy.data.cameras.new("VerifyCam")
    cam = bpy.data.objects.new("VerifyCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam.location = (0.0, 0.0, 25.0)
    cam.rotation_euler = (math.pi * 0.5, 0.0, 0.0)  # Horizontal equirectangular (+Z up)
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 512
    scene.cycles.samples = 12
    pano_path = os.path.join(REVIEW_DIR, "cp7a_hdri_sun_pano.jpg")
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 88
    scene.render.filepath = pano_path
    bpy.ops.render.render(write_still=True)

    # Measure rendered panorama brightest sky pixel vs configured SUN_AZIMUTH_DEG
    pano_im = cv2.imread(pano_path)
    pano_gray = cv2.cvtColor(pano_im, cv2.COLOR_BGR2GRAY)
    upper = pano_gray[:pano_gray.shape[0] // 2, :]
    py_max, px_max = np.unravel_index(np.argmax(upper), upper.shape)
    # In Blender PANO camera rotated (pi/2, 0, 0): center column faces +Y (az=90), left is +X (az=0), right is -X (az=180)
    pano_az = (90.0 - ((px_max / pano_gray.shape[1]) - 0.5) * 360.0) % 360.0
    pano_el = (0.5 - py_max / pano_gray.shape[0]) * 180.0
    az_diff = abs(((pano_az - sun_meta["target_sun_az_deg"] + 180.0) % 360.0) - 180.0)
    sun_meta["pano_measured_az_deg"] = round(pano_az, 2)
    sun_meta["pano_measured_el_deg"] = round(pano_el, 2)
    sun_meta["sun_hdri_az_error_deg"] = round(az_diff, 2)
    print(f"[HDRI-SUN CHECK] Target az={sun_meta['target_sun_az_deg']} deg, Pano peak az={pano_az:.1f} deg, error={az_diff:.1f} deg")

    # Reset camera to PERSPECTIVE
    cam_data.type = "PERSP"
    cam_data.sensor_width = 36.0

    # =========================================================================
    # B. OPENCV SOLVEPNP SILHOUETTE VERIFICATION AGAINST 2 REAL PHOTOS
    # =========================================================================
    # 3D landmark points on our L0-950 train (in train local coords: nose tip at X=0, extends -X, +Y left, +Z up)
    # Right side (-Y) is visible in both L0-950.jpg and Yamanashi_Maglev_Test_Track5.jpg!
    _, zb0, zt0, _, _ = l0_nose_envelope(0.2)
    _, _, zt_vis, _, _ = l0_nose_envelope(10.4)
    _, _, zt_15, _, _ = l0_nose_envelope(15.0)

    # Ref 1: L0-950.jpg (1251 x 704) — train heading right-down (+X), camera on right-side (-Y) elevated
    # Exact manually verified pixel coordinates on L0-950.jpg (1251x704):
    ref1_path = os.path.join(SCRATCH_REFS, "L0-950.jpg")
    pts3d_1 = np.array([
        [0.0,    0.0,  0.5 * (zb0 + zt0)],  # 0: Front nose tip center
        [-10.4,  0.0,  zt_vis],              # 1: High camera visor center
        [-15.0,  0.0,  specs.CAR_TOP_Z],     # 2: Roof crown behind nose (s=15m)
        [-28.0,  0.0,  specs.CAR_TOP_Z],     # 3: Car 1-2 roof joint (s=28m)
        [-52.3,  0.0,  specs.CAR_TOP_Z],     # 4: Car 2-3 roof joint (s=52.3m)
        [-76.6,  0.0,  specs.CAR_TOP_Z],     # 5: Car 3-4 roof joint (s=76.6m)
        [-28.0, -1.45, specs.WINDOW_Z_CENTER],  # 6: Car 1-2 lower-right body joint
        [-52.3, -1.45, specs.WINDOW_Z_CENTER],  # 7: Car 2-3 lower-right body joint
        [-17.0, -1.45, specs.WINDOW_Z_CENTER],  # 8: End car first window
    ], dtype=np.float64)
    pts2d_1 = np.array([
        [1224.0, 650.0],  # 0: Nose tip
        [988.0,  479.0],  # 1: Camera visor center
        [868.0,  434.0],  # 2: Roof crown s=15m
        [715.0,  421.0],  # 3: Car 1-2 roof joint
        [523.0,  376.0],  # 4: Car 2-3 roof joint
        [360.0,  330.0],  # 5: Car 3-4 roof joint
        [707.0,  488.0],  # 6: Car 1-2 side mid
        [516.0,  426.0],  # 7: Car 2-3 side mid
        [818.0,  495.0],  # 8: End car first window
    ], dtype=np.float64)

    # Ref 2: Yamanashi_Maglev_Test_Track5.jpg (1280 x 827) — elevated right-side (-Y) view of nose & 5 cars
    ref2_path = os.path.join(SCRATCH_REFS, "Yamanashi_Maglev_Test_Track5.jpg")
    pts3d_2 = np.array([
        [0.0,    0.0,  0.5 * (zb0 + zt0)],  # 0: Nose tip
        [-10.4,  0.0,  zt_vis],              # 1: Mid-nose roof slope (s=10.4m)
        [-15.0,  0.0,  specs.CAR_TOP_Z],     # 2: Roof crown (s=15m)
        [-28.0,  0.0,  specs.CAR_TOP_Z],     # 3: Car 1-2 roof joint
        [-52.3,  0.0,  specs.CAR_TOP_Z],     # 4: Car 2-3 roof joint
        [-76.6,  0.0,  specs.CAR_TOP_Z],     # 5: Car 3-4 roof joint
        [-28.0, -1.45, specs.WINDOW_Z_CENTER],  # 6: Car 1-2 side
        [-52.3, -1.45, specs.WINDOW_Z_CENTER],  # 7: Car 2-3 side
    ], dtype=np.float64)
    pts2d_2 = np.array([
        [1082.0, 675.0],  # 0: Nose tip
        [835.0,  496.0],  # 1: Nose slope s=10.4m
        [756.0,  445.0],  # 2: Roof crown s=15m
        [585.0,  416.0],  # 3: Car 1-2 roof joint
        [406.0,  374.0],  # 4: Car 2-3 roof joint
        [278.0,  337.0],  # 5: Car 3-4 roof joint
        [575.0,  474.0],  # 6: Car 1-2 side
        [402.0,  420.0],  # 7: Car 2-3 side
    ], dtype=np.float64)

    pnp_results = {}
    for tag, rpath, p3, p2, fov_g in [
        ("ref1_L0_950", ref1_path, pts3d_1, pts2d_1, 18.0),
        ("ref2_Yamanashi5", ref2_path, pts3d_2, pts2d_2, 24.0),
    ]:
        pnp = solve_pnp_for_photo(rpath, p3, p2, fov_g)
        set_blender_cam_from_pnp(cam, pnp)
        scene.render.resolution_x = pnp["W"]
        scene.render.resolution_y = pnp["H"]
        scene.cycles.samples = 28
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        # Hide guideway briefly so silhouette alpha of ONLY the train is crisp for overlay!
        gw_objs = [o for o in scene.collection.objects if o.name.startswith(("Guideway", "Pier", "Mast"))]
        for o in gw_objs:
            o.hide_render = True
        cgi_png = os.path.join(ARTIFACT_DIR, f"scratch_{tag}_cgi.png")
        scene.render.filepath = cgi_png
        bpy.ops.render.render(write_still=True)
        for o in gw_objs:
            o.hide_render = False
        scene.render.film_transparent = False

        # Compose 50/50 overlay + green CGI silhouette contour + PnP landmark dots (saved to artifact dir, NOT repo!)
        ref_bgr = cv2.imread(rpath)
        cgi_bgra = cv2.imread(cgi_png, cv2.IMREAD_UNCHANGED)
        alpha = cgi_bgra[:, :, 3:4].astype(np.float32) / 255.0
        cgi_bgr = cgi_bgra[:, :, :3].astype(np.float32)
        blend = (ref_bgr.astype(np.float32) * (1.0 - 0.50 * alpha) + cgi_bgr * (0.50 * alpha)).astype(np.uint8)
        # Draw crisp bright-green silhouette contour of CGI train over real photo
        mask_u8 = (cgi_bgra[:, :, 3] > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, contours, -1, (40, 255, 80), 2)
        # Draw target (red) vs reprojected (cyan) PnP points
        for (tx, ty), (px, py) in zip(p2, pnp["proj"]):
            cv2.circle(blend, (int(tx), int(ty)), 6, (0, 0, 255), 2)
            cv2.circle(blend, (int(px), int(py)), 4, (255, 255, 0), -1)
        cv2.putText(
            blend,
            f"solvePnP {tag} | lens={cam.data.lens:.1f}mm | RMS={pnp['rms_px']:.2f}px (max={pnp['max_px']:.2f}px)",
            (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA
        )
        ov_path = os.path.join(ARTIFACT_DIR, f"cp7a_pnp_overlay_{tag}.jpg")
        cv2.imwrite(ov_path, blend, [cv2.IMWRITE_JPEG_QUALITY, 90])

        pnp_results[tag] = {
            "image_size": [pnp["W"], pnp["H"]],
            "solved_lens_mm_eq36": round(float(cam.data.lens), 2),
            "rms_residual_px": round(pnp["rms_px"], 2),
            "max_residual_px": round(pnp["max_px"], 2),
            "per_point_px": pnp["per_point_px"],
            "overlay_artifact": ov_path,
        }
        print(f"[solvePnP {tag}] lens={cam.data.lens:.1f}mm, RMS={pnp['rms_px']:.2f} px, max={pnp['max_px']:.2f} px")

    # =========================================================================
    # C. CHECKPOINT 7a HERO TRAIN INSPECTION STILLS (COMMITTED TO review/)
    # =========================================================================
    # Place 4 m metric ruler (with red 10 cm bottom levitation-gap marker) right inside U-trough at x=-6.4 m (lead bogie)!
    ruler = add_scale_ruler(scene.collection, x_pos=-6.4, y_pos=-1.48, z_base=0.0)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.cycles.samples = 36
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90

    stills = [
        ("cp7a_train_hero_34.jpg", (22.0, -14.0, 5.8), (-12.0, 0.0, 1.8), 65.0, False),
        ("cp7a_train_levitation_ruler.jpg", (7.5, -1.48, 0.42), (-6.4, 0.0, 0.55), 55.0, True),
        ("cp7a_train_windows_interior.jpg", (-35.0, -4.6, 2.35), (-38.5, 0.0, 2.02), 70.0, False),
    ]
    for fname, loc_c, target, lens_mm, show_ruler in stills:
        ruler.hide_render = not show_ruler
        cam.location = loc_c
        direction = Vector(target) - Vector(loc_c)
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = lens_mm
        out_p = os.path.join(REVIEW_DIR, fname)
        scene.render.filepath = out_p
        bpy.ops.render.render(write_still=True)
        # Also copy to artifact dir for direct viewing
        Image.open(out_p).save(os.path.join(ARTIFACT_DIR, fname), quality=92)
        print(f"[7a still] Rendered {out_p}")

    report = {"sun_hdri_alignment": sun_meta, "solve_pnp": pnp_results}
    with open(os.path.join(REVIEW_DIR, "cp7a_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify_pnp_and_render_checkpoint()
