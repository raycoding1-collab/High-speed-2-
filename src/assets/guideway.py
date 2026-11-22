"""
Authentic JR Central SCMaglev U-shaped concrete guideway, figure-8 sidewall ground coil panels,
haunched box-girder viaduct with arched double-column piers, steel inspection walkways,
slotted micro-pressure-wave tunnel portal buffer hood, and ribbed acoustic noise-reduction hood.

All dimensions strictly follow `src/specs.py`.
"""
import math
import os
import bpy
import bmesh
import numpy as np
from PIL import Image, ImageDraw
from mathutils import Vector, Matrix, Euler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import specs


# ==============================================================================
# 1. PROCEDURAL TEXTURES (FIGURE-8 GROUND COILS + WEATHERED JAPANESE CONCRETE)
# ==============================================================================
def generate_guideway_textures(tex_dir: str) -> dict:
    os.makedirs(tex_dir, exist_ok=True)
    rng = np.random.default_rng(108)

    # A) White FRP sidewall ground coil panels (4 coil modules = 3.60 m wide x 1.12 m tall)
    coil_alb = os.path.join(tex_dir, "scmaglev_coils_albedo.png")
    coil_bmp = os.path.join(tex_dir, "scmaglev_coils_bump.png")
    W, H = 2048, 640
    img_a = Image.new("RGB", (W, H), (218, 220, 218))
    img_b = Image.new("L", (W, H), 150)
    da = ImageDraw.Draw(img_a)
    db = ImageDraw.Draw(img_b)

    n_mod = 4
    mw = W // n_mod
    for i in range(n_mod):
        x0 = i * mw
        # Vertical precast panel joint
        da.rectangle([x0, 0, x0 + 6, H], fill=(108, 112, 116))
        db.rectangle([x0, 0, x0 + 6, H], fill=65)
        # Upper and lower figure-8 rounded-rectangle coil loops
        for cy in (int(H * 0.28), int(H * 0.72)):
            rx0, rx1 = x0 + 38, x0 + mw - 38
            ry0, ry1 = cy - 110, cy + 110
            # Recessed outline
            da.rounded_rectangle([rx0, ry0, rx1, ry1], radius=48, outline=(172, 176, 178), width=12)
            db.rounded_rectangle([rx0, ry0, rx1, ry1], radius=48, outline=115, width=14)
            # Dark mounting bolt recesses (unmistakable repeating dot array in Yamanashi photos!)
            for bx in (x0 + 75, x0 + mw // 2, x0 + mw - 75):
                for by in (cy - 52, cy + 52):
                    da.ellipse([bx - 11, by - 11, bx + 11, by + 11], fill=(52, 56, 60))
                    db.ellipse([bx - 12, by - 12, bx + 12, by + 12], fill=55)
        # Bottom drainage slot & dark grime strip
        da.rectangle([x0, H - 48, x0 + mw, H], fill=(142, 140, 134))

    # Subtle grime & rain variation
    arr_a = np.array(img_a, dtype=np.float32)
    noise = rng.normal(0, 5.5, (H, W, 1)).astype(np.float32)
    # Vertical rain streaks
    streak = rng.normal(0, 6.0, (1, W, 1)).astype(np.float32)
    arr_a = np.clip(arr_a + noise + streak * np.linspace(1.0, 0.3, H)[:, None, None], 0, 255).astype(np.uint8)
    Image.fromarray(arr_a).save(coil_alb)
    img_b.save(coil_bmp)

    # B) Weathered Japanese Viaduct Concrete (formwork joints, tie-rod holes, rainwater streaks, efflorescence)
    conc_alb = os.path.join(tex_dir, "viaduct_concrete_albedo.png")
    conc_rgh = os.path.join(tex_dir, "viaduct_concrete_rough.png")
    conc_bmp = os.path.join(tex_dir, "viaduct_concrete_bump.png")
    CW, CH = 2048, 1024
    base = np.full((CH, CW, 3), [148, 146, 140], dtype=np.float32)

    # Multi-scale concrete mottling
    for scale, amp in [(32, 12.0), (128, 7.0), (512, 4.5)]:
        n = rng.normal(0.0, amp, (max(4, CH // scale), max(4, CW // scale))).astype(np.float32)
        ni = Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8)).resize((CW, CH), Image.BILINEAR)
        base += (np.array(ni, dtype=np.float32) - 128.0)[:, :, None]

    # Vertical precast parapet/girder panel joints (every 256 px = ~2.5m)
    bmp_c = np.full((CH, CW), 135, dtype=np.float32)
    for jx in range(0, CW, 256):
        # Slight panel-to-panel tonal shift
        shift = rng.uniform(-7.0, 7.0)
        base[:, jx:min(CW, jx + 256), :] += shift
        base[:, jx:min(CW, jx + 5), :] -= 38.0
        bmp_c[:, jx:min(CW, jx + 5)] = 70.0
        # Dark rain runoff drip from top coping at panel joints
        drip_h = int(rng.uniform(CH * 0.35, CH * 0.85))
        grad = np.linspace(1.0, 0.0, drip_h)[:, None, None]
        base[:drip_h, max(0, jx - 14):min(CW, jx + 18), :] -= 22.0 * grad

    # General vertical rain streaks from top edge + pale white efflorescence patches
    streaks = rng.normal(0, 1.0, (1, CW // 4)).astype(np.float32)
    si = np.array(Image.fromarray(np.clip(streaks * 40 + 128, 0, 255).astype(np.uint8)).resize((CW, CH), Image.BILINEAR), dtype=np.float32) - 128.0
    v_fade = np.linspace(1.0, 0.25, CH)[:, None]
    base -= np.clip(si * v_fade * 0.35, -10.0, 22.0)[:, :, None]

    # Formwork tie-rod holes grid
    for ty in (int(CH * 0.28), int(CH * 0.68)):
        for tx in range(64, CW, 128):
            base[ty - 4:ty + 5, tx - 4:tx + 5] -= 45.0
            bmp_c[ty - 4:ty + 5, tx - 4:tx + 5] = 65.0

    # Roughness: concrete ~0.72..0.92
    rgh_c = np.clip(205.0 - (base[:, :, 0] - 145.0) * 0.65, 165, 240).astype(np.uint8)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(conc_alb)
    Image.fromarray(rgh_c).save(conc_rgh)
    Image.fromarray(np.clip(bmp_c, 0, 255).astype(np.uint8)).save(conc_bmp)

    return {
        "coil_albedo": coil_alb,
        "coil_bump": coil_bmp,
        "conc_albedo": conc_alb,
        "conc_rough": conc_rgh,
        "conc_bump": conc_bmp,
    }


def _load_img(path: str, colorspace: str = "sRGB"):
    name = os.path.basename(path)
    if name in bpy.data.images:
        img = bpy.data.images[name]
        img.reload()
    else:
        img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    return img


def make_guideway_materials(tex_dir: str) -> dict:
    from assets.pbr_materials import make_box_pbr_material
    t = generate_guideway_textures(tex_dir)
    mats = {}

    # 1) Real photogrammetry CC0 Poly Haven weathered concrete (Box-projected -> zero UV stretching on piers, girders & portals!)
    mats["concrete"] = make_box_pbr_material(
        "Mat_Viaduct_Concrete", "concrete", tile_scale=0.14,
        tint_rgb=(0.94, 0.92, 0.88), tint_fac=0.35, normal_strength=0.9,
        rough_mult=1.05, ao_darkening=0.52, ao_dist=2.6, macro_var=0.22
    )

    # 2) Sidewall figure-8 coil panels
    if "Mat_SCMaglev_Coils" in bpy.data.materials:
        mats["coils"] = bpy.data.materials["Mat_SCMaglev_Coils"]
    else:
        m = bpy.data.materials.new("Mat_SCMaglev_Coils")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.42
        ta = nt.nodes.new("ShaderNodeTexImage")
        ta.image = _load_img(t["coil_albedo"], "sRGB")
        tb = nt.nodes.new("ShaderNodeTexImage")
        tb.image = _load_img(t["coil_bump"], "Non-Color")
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = 0.48
        bmp.inputs["Distance"].default_value = 0.03
        nt.links.new(ta.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tb.outputs["Color"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        mats["coils"] = m

    # 3) Trough floor concrete (real photogrammetry concrete, slightly darker weathered rubber-tire path)
    mats["floor"] = make_box_pbr_material(
        "Mat_Trough_Floor", "concrete", tile_scale=0.22,
        tint_rgb=(0.72, 0.71, 0.68), tint_fac=0.55, normal_strength=0.75,
        rough_mult=1.1, ao_darkening=0.45, ao_dist=1.5, macro_var=0.16
    )

    # 4) Galvanized steel (railings, lightning masts, hood ribs)
    if "Mat_Galv_Steel" in bpy.data.materials:
        mats["steel"] = bpy.data.materials["Mat_Galv_Steel"]
    else:
        m = bpy.data.materials.new("Mat_Galv_Steel")
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.36, 0.38, 0.40, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.82
        bsdf.inputs["Roughness"].default_value = 0.38
        mats["steel"] = m

    # 5) Noise-reduction hood arched acoustic panels
    if "Mat_NoiseHood_Roof" in bpy.data.materials:
        mats["hood_roof"] = bpy.data.materials["Mat_NoiseHood_Roof"]
    else:
        m = bpy.data.materials.new("Mat_NoiseHood_Roof")
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.42, 0.39, 0.35, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.32
        bsdf.inputs["Roughness"].default_value = 0.45
        mats["hood_roof"] = m

    return mats


# ==============================================================================
# 2. EXTRUDED DOUBLE-TRACK VIADUCT ALONG EXACT GUIDEWAY CURVE / CANT / GRADE
# ==============================================================================
def _swept_profile(bm: bmesh.types.BMesh, uv_layer, profile_yz: list, s_vals: list,
                   loc: specs.LocationSpec, u_scale: float = 0.1, v_scale: float = 0.25,
                   mat_idx: int = 0, closed: bool = True):
    """Sweep a 2D (y, z) cross-section along `guideway_pose_at_s(s, loc)` with exact curve & cant."""
    n_p = len(profile_yz)
    # Compute cumulative perimeter for clean V UV
    v_coords = [0.0]
    for i in range(1, n_p):
        dy = profile_yz[i][0] - profile_yz[i - 1][0]
        dz = profile_yz[i][1] - profile_yz[i - 1][1]
        v_coords.append(v_coords[-1] + math.hypot(dy, dz) * v_scale)

    rings = []
    for s in s_vals:
        pos, eul = specs.guideway_pose_at_s(s, loc)
        mat = Matrix.Translation(pos) @ Euler(eul, "XYZ").to_matrix().to_4x4()
        ring = [bm.verts.new(mat @ Vector((0.0, y, z))) for (y, z) in profile_yz]
        rings.append((s, ring))

    n_seg = n_p if closed else n_p - 1
    for i in range(len(rings) - 1):
        sa, ra = rings[i]
        sb, rb = rings[i + 1]
        ua = sa * u_scale
        ub = sb * u_scale
        for j in range(n_seg):
            jn = (j + 1) % n_p
            f = bm.faces.new((ra[j], ra[jn], rb[jn], rb[j]))
            f.material_index = mat_idx
            va = v_coords[j]
            vb = v_coords[jn] if jn > j else v_coords[-1] + 0.2
            for lp, uv in zip(f.loops, [(ua, va), (ua, vb), (ub, vb), (ub, va)]):
                lp[uv_layer].uv = uv


def build_guideway_and_viaduct(
    collection: bpy.types.Collection,
    loc: specs.LocationSpec,
    tex_dir: str,
    s_start: float = -700.0,
    s_end: float = 1200.0,
    z_offset: float = 0.0,
    pier_height: float = None,
    add_portal_at_s: float = None,
    add_exit_portal_at_s: float = None,
    noise_hood_range: tuple = None,
) -> list:
    """
    Build the complete double-track SCMaglev guideway + box girder + piers + lightning masts
    + optional tunnel buffer hood(s) & acoustic noise-reduction hood.
    Active train runs on Track 1 centered at local Y = 0.0; opposing Track 2 sits at Y = +5.80 m
    (so double-track centerline is at Y = +2.90 m).
    """
    mats = make_guideway_materials(tex_dir)
    if pier_height is None:
        pier_height = loc.viaduct_elev_above_ground

    # Shift everything vertically by z_offset via a parent empty so guideway floor at s=0 is at Z=z_offset
    root = bpy.data.objects.new(f"Guideway_{loc.loc_id}", None)
    root.location = (0.0, 0.0, z_offset)
    collection.objects.link(root)

    n_steps = max(40, int((s_end - s_start) / 20.0))
    s_vals = list(np.linspace(s_start, s_end, n_steps))

    # Double-track center is y_mid = +TRACK_SPACING/2 = +2.90 m (Track 1 at y=0, Track 2 at y=+5.80 m)
    yc = specs.TRACK_SPACING * 0.5  # 2.90 m
    half_deck = specs.VIADUCT_DECK_WIDTH * 0.5  # 6.30 m
    yL = yc + half_deck  # +9.20 m
    yR = yc - half_deck  # -3.40 m

    # 1) Main Concrete Viaduct Deck, Box Girder & Sidewalls (material 0=concrete, 1=coils, 2=floor)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")

    # Haunched box-girder underside + outer deck slabs (authentic 1.15 m vertical concrete fascia!)
    girder_prof = [
        (yR, 0.0),
        (yR, -1.15),
        (yc - 3.6, -1.65),
        (yc - 3.2, -specs.VIADUCT_GIRDER_DEPTH),
        (yc + 3.2, -specs.VIADUCT_GIRDER_DEPTH),
        (yc + 3.6, -1.65),
        (yL, -1.15),
        (yL, 0.0),
    ]
    _swept_profile(bm, uv, girder_prof, s_vals, loc, u_scale=0.05, v_scale=0.18, mat_idx=0, closed=True)

    # Outer acoustic parapet walls (left and right edges, extending cleanly into girder deck)
    ph = specs.VIADUCT_PARAPET_HEIGHT
    for py0, py1 in [(yR, yR + 0.42), (yL - 0.42, yL)]:
        p_prof = [(py0, -0.62), (py1, -0.62), (py1, ph), (py0, ph)]
        _swept_profile(bm, uv, p_prof, s_vals, loc, u_scale=0.05, v_scale=0.25, mat_idx=0, closed=True)

    # U-Trough Concrete Sidewalls & Floor for Track 1 (y=0.0) AND Track 2 (y=+5.80 m)
    hw_tr = specs.TROUGH_INNER_WIDTH * 0.5  # 1.70 m
    sw_h = specs.SIDEWALL_HEIGHT            # 1.68 m
    sw_t = specs.SIDEWALL_THICKNESS         # 0.48 m
    for ty in (0.0, specs.TRACK_SPACING):
        # Trough floor slab (raised +0.015 m cleanly above box girder z=0 to prevent z-fighting!)
        fl_prof = [(ty - hw_tr, 0.015), (ty + hw_tr, 0.015), (ty + hw_tr, -0.05), (ty - hw_tr, -0.05)]
        _swept_profile(bm, uv, fl_prof, s_vals, loc, u_scale=0.1, v_scale=0.2, mat_idx=2, closed=True)

        # Left & right structural concrete sidewalls
        for sy0, sy1 in [(ty - hw_tr - sw_t, ty - hw_tr), (ty + hw_tr, ty + hw_tr + sw_t)]:
            sw_prof = [(sy0, -0.05), (sy1, -0.05), (sy1, sw_h), (sy0, sw_h)]
            _swept_profile(bm, uv, sw_prof, s_vals, loc, u_scale=0.08, v_scale=0.3, mat_idx=0, closed=True)

        # Inner white FRP figure-8 ground coil panels lining BOTH inner faces of each U-trough!
        cz0 = specs.COIL_PANEL_Z_BOTTOM
        cz1 = cz0 + specs.COIL_PANEL_HEIGHT
        # U scale = 1 / (4 * 0.90 m) = 1 / 3.60 m so 4 coil modules repeat every 3.60 m!
        u_coil = 1.0 / (4.0 * specs.COIL_PANEL_PITCH)
        v_coil = 1.0 / specs.COIL_PANEL_HEIGHT
        # Right inner face (facing +Y)
        c_prof_r = [(ty - hw_tr + 0.12, cz0), (ty - hw_tr + 0.12, cz1)]
        _swept_profile(bm, uv, c_prof_r, s_vals, loc, u_scale=u_coil, v_scale=v_coil, mat_idx=1, closed=False)
        # Left inner face (facing -Y)
        c_prof_l = [(ty + hw_tr - 0.12, cz1), (ty + hw_tr - 0.12, cz0)]
        _swept_profile(bm, uv, c_prof_l, s_vals, loc, u_scale=u_coil, v_scale=v_coil, mat_idx=1, closed=False)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh_gw = bpy.data.meshes.new(f"GuidewayDeck_{loc.loc_id}")
    bm.to_mesh(mesh_gw)
    bm.free()
    mesh_gw.materials.append(mats["concrete"])
    mesh_gw.materials.append(mats["coils"])
    mesh_gw.materials.append(mats["floor"])
    obj_gw = bpy.data.objects.new(f"GuidewayDeck_{loc.loc_id}", mesh_gw)
    obj_gw.parent = root
    collection.objects.link(obj_gw)

    # 2) Arched Double-Column Concrete Piers + Steel Inspection Walkways + Lightning Masts (instanced!)
    bm_pier = bmesh.new()
    # Pier cap beam (under box girder)
    p_top = -specs.VIADUCT_GIRDER_DEPTH
    p_bot = -pier_height - 14.0  # Extend safely into sloped DEM terrain so no pier ever floats!
    p_h = p_top - p_bot
    bmesh.ops.create_cube(bm_pier, size=1.0,
                          matrix=Matrix.Translation((0.0, yc, p_top - 1.1)) @ Matrix.Diagonal((2.8, 9.6, 2.2, 1.0)))
    # Twin vertical columns with authentic Japanese Shinkansen/SCMaglev arched haunches & tapered footing
    for dy in (-3.1, 3.1):
        bmesh.ops.create_cube(bm_pier, size=1.0,
                              matrix=Matrix.Translation((0.0, yc + dy, p_top - 1.8 - p_h * 0.5)) @ Matrix.Diagonal((2.3, 2.2, p_h, 1.0)))
        # Inner arched haunch bracket joining column to cap beam (matches real Yamanashi viaduct photos!)
        sign_y = -1.0 if dy > 0 else 1.0
        haunch_m = (
            Matrix.Translation((0.0, yc + dy + sign_y * 1.45, p_top - 2.65))
            @ Matrix.Rotation(sign_y * math.radians(32.0), 4, "X")
            @ Matrix.Diagonal((2.2, 1.9, 1.4, 1.0))
        )
        bmesh.ops.create_cube(bm_pier, size=1.0, matrix=haunch_m)
    # Transverse tie beam halfway down tall piers
    if pier_height > 20.0:
        bmesh.ops.create_cube(bm_pier, size=1.0,
                              matrix=Matrix.Translation((0.0, yc, p_top - pier_height * 0.52)) @ Matrix.Diagonal((1.6, 6.2, 1.6, 1.0)))
    mesh_pier = bpy.data.meshes.new(f"ViaductPier_{loc.loc_id}")
    bm_pier.to_mesh(mesh_pier)
    bm_pier.free()
    mesh_pier.materials.append(mats["concrete"])

    # Lightning arrestor masts (slender vertical grey poles along outer parapet — NO catenary!)
    bm_mast = bmesh.new()
    for py in (yR + 0.21, yL - 0.21):
        bmesh.ops.create_cone(bm_mast, cap_ends=True, segments=10, radius1=0.09, radius2=0.055,
                              depth=specs.LIGHTNING_MAST_HEIGHT,
                              matrix=Matrix.Translation((0.0, py, ph + specs.LIGHTNING_MAST_HEIGHT * 0.5)))
    # Steel inspection platform around pier cap
    bmesh.ops.create_cube(bm_mast, size=1.0,
                          matrix=Matrix.Translation((0.0, yc, p_top - 1.9)) @ Matrix.Diagonal((4.0, 10.6, 0.12, 1.0)))
    mesh_mast = bpy.data.meshes.new(f"ViaductMasts_{loc.loc_id}")
    bm_mast.to_mesh(mesh_mast)
    bm_mast.free()
    mesh_mast.materials.append(mats["steel"])

    created = [root, obj_gw]
    s_p = math.ceil(s_start / specs.VIADUCT_PIER_SPACING) * specs.VIADUCT_PIER_SPACING
    p_idx = 0
    while s_p <= s_end:
        # Skip piers inside mountain tunnels
        in_portal = False
        if add_portal_at_s is not None and s_p < add_portal_at_s - 4.0:
            in_portal = True
        if add_exit_portal_at_s is not None and s_p > add_exit_portal_at_s + 4.0:
            in_portal = True
        if not in_portal:
            pos, eul = specs.guideway_pose_at_s(s_p, loc)
            pier_obj = bpy.data.objects.new(f"Pier_{loc.loc_id}_{p_idx:03d}", mesh_pier)
            pier_obj.parent = root
            pier_obj.location = pos
            pier_obj.rotation_euler = (0.0, 0.0, eul[2])  # Piers stay plumb vertical!
            collection.objects.link(pier_obj)

            mast_obj = bpy.data.objects.new(f"Mast_{loc.loc_id}_{p_idx:03d}", mesh_mast)
            mast_obj.parent = root
            mast_obj.location = pos
            mast_obj.rotation_euler = eul
            collection.objects.link(mast_obj)
            created.extend([pier_obj, mast_obj])
            p_idx += 1
        s_p += specs.VIADUCT_PIER_SPACING

    # 3) Optional Slotted Tunnel Portal Micro-Pressure-Wave Buffer Hood(s)
    for tag, ps, flip in [("Entry", add_portal_at_s, False), ("Exit", add_exit_portal_at_s, True)]:
        if ps is None:
            continue
        hood_obj = _build_buffer_hood(f"BufferHood_{loc.loc_id}_{tag}", loc, ps, flip, mats)
        hood_obj.parent = root
        collection.objects.link(hood_obj)
        created.append(hood_obj)

    # 4) Optional Acoustic Noise-Reduction Enclosure Hood (near rural houses)
    if noise_hood_range is not None:
        nh_obj = _build_noise_hood(f"NoiseHood_{loc.loc_id}", loc, noise_hood_range[0], noise_hood_range[1], mats)
        nh_obj.parent = root
        collection.objects.link(nh_obj)
        created.append(nh_obj)

    return created


def _build_buffer_hood(name: str, loc: specs.LocationSpec, s_mouth: float, flip: bool, mats: dict) -> bpy.types.Object:
    """
    Build a 56 m slotted reinforced-concrete micro-pressure-wave tunnel buffer hood
    plus dark deep mountain tunnel bore behind it.
    If flip=False: mouth opens at s_mouth facing +X (train exits into open air toward +X).
    If flip=True:  mouth opens at s_mouth facing -X (train enters tunnel traveling +X).
    """
    bm = bmesh.new()
    yc = specs.TRACK_SPACING * 0.5
    W = specs.BUFFER_HOOD_WIDTH
    H = specs.BUFFER_HOOD_HEIGHT
    L = specs.BUFFER_HOOD_LENGTH
    wt = 0.75

    dir_sign = -1.0 if not flip else 1.0  # Direction from mouth INTO the mountain

    # Build hood in segments with pressure-relief ventilation slits along upper walls & roof
    n_slits = specs.BUFFER_HOOD_SLITS
    seg_len = L / (n_slits * 2 + 1)
    for k in range(n_slits * 2 + 1):
        x0 = dir_sign * (k * seg_len)
        x1 = dir_sign * ((k + 1) * seg_len)
        xc = 0.5 * (x0 + x1)
        is_slit = (k % 2 == 1)

        # Lower solid concrete sidewalls (always solid)
        for sy in (yc - W * 0.5, yc + W * 0.5):
            h_wall = H * 0.55 if is_slit else H
            bmesh.ops.create_cube(bm, size=1.0,
                                  matrix=Matrix.Translation((xc, sy, h_wall * 0.5)) @ Matrix.Diagonal((seg_len + 0.05, wt, h_wall, 1.0)))
        if not is_slit:
            # Solid concrete arch/box roof rib
            bmesh.ops.create_cube(bm, size=1.0,
                                  matrix=Matrix.Translation((xc, yc, H - wt * 0.5)) @ Matrix.Diagonal((seg_len + 0.05, W + wt, wt, 1.0)))
        else:
            # Slender roof longitudinal beams flanking roof vent slot
            for sy in (yc - W * 0.34, yc + W * 0.34):
                bmesh.ops.create_cube(bm, size=1.0,
                                      matrix=Matrix.Translation((xc, sy, H - wt * 0.5)) @ Matrix.Diagonal((seg_len + 0.05, 1.8, wt, 1.0)))

    # Massive portal headwall & deep dark tunnel bore behind the buffer hood (300 m long)
    bore_x = dir_sign * (L + 150.0)
    # Tapered headwall collar embedded deeply into mountain slope (no abrupt square block sticking out!)
    bmesh.ops.create_cube(bm, size=1.0,
                          matrix=Matrix.Translation((dir_sign * (L + 1.5), yc, H + 1.6)) @ Matrix.Diagonal((5.5, W + 6.0, 4.0, 1.0)))
    # Sloped outer concrete wingwalls angled back into the hillside terrain
    for side in (-1.0, 1.0):
        wy = yc + side * (W * 0.5 + 3.2)
        wm = (
            Matrix.Translation((dir_sign * (L + 0.8), wy, H * 0.42))
            @ Matrix.Rotation(side * dir_sign * math.radians(24.0), 4, "Z")
            @ Matrix.Diagonal((7.5, 2.2, H * 0.92, 1.0))
        )
        bmesh.ops.create_cube(bm, size=1.0, matrix=wm)
    # Dark interior tunnel liner (walls + roof)
    for sy in (yc - W * 0.48, yc + W * 0.48):
        bmesh.ops.create_cube(bm, size=1.0,
                              matrix=Matrix.Translation((bore_x, sy, H * 0.5)) @ Matrix.Diagonal((300.0, 0.8, H, 1.0)))
    bmesh.ops.create_cube(bm, size=1.0,
                          matrix=Matrix.Translation((bore_x, yc, H)) @ Matrix.Diagonal((300.0, W, 0.8, 1.0)))
    # Back light-blocker cap inside tunnel so no sky leaks through!
    bmesh.ops.create_cube(bm, size=1.0,
                          matrix=Matrix.Translation((dir_sign * (L + 298.0), yc, H * 0.5)) @ Matrix.Diagonal((1.0, W, H, 1.0)))

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mats["concrete"])

    pos, eul = specs.guideway_pose_at_s(s_mouth, loc)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = pos
    obj.rotation_euler = eul
    return obj


def _build_noise_hood(name: str, loc: specs.LocationSpec, s0: float, s1: float, mats: dict) -> bpy.types.Object:
    """Build a ribbed arched acoustic noise-reduction enclosure hood over viaduct near residential houses."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    yc = specs.TRACK_SPACING * 0.5
    hw = specs.NOISE_HOOD_WIDTH * 0.5
    H = specs.NOISE_HOOD_HEIGHT

    # Arched canopy profile above parapet walls
    arch_pts = []
    for i in range(15):
        ang = math.pi * i / 14.0  # 0 (+Y side) to pi (-Y side)
        y = yc + hw * math.cos(ang)
        z = specs.VIADUCT_PARAPET_HEIGHT + (H - specs.VIADUCT_PARAPET_HEIGHT) * math.sin(ang)
        arch_pts.append((y, z))

    s_vals = list(np.linspace(s0, s1, max(16, int((s1 - s0) / 10.0))))
    _swept_profile(bm, uv, arch_pts, s_vals, loc, u_scale=0.15, v_scale=0.15, mat_idx=0, closed=False)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.22)

    # Structural exterior steel ribs every 6 m
    s_r = s0
    while s_r <= s1:
        pos, eul = specs.guideway_pose_at_s(s_r, loc)
        m = Matrix.Translation(pos) @ Euler(eul, "XYZ").to_matrix().to_4x4()
        for k in range(len(arch_pts) - 1):
            ya, za = arch_pts[k]
            yb, zb = arch_pts[k + 1]
            mid = Vector((0.0, 0.5 * (ya + yb), 0.5 * (za + zb) + 0.14))
            seg = math.hypot(yb - ya, zb - za)
            ang = math.atan2(zb - za, yb - ya)
            rm = m @ Matrix.Translation(mid) @ Matrix.Rotation(ang, 4, "X") @ Matrix.Diagonal((0.28, seg + 0.05, 0.24, 1.0))
            ret = bmesh.ops.create_cube(bm, size=1.0, matrix=rm)
            for v in ret["verts"]:
                for f in v.link_faces:
                    f.material_index = 1
        s_r += 6.0

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mats["hood_roof"])
    mesh.materials.append(mats["steel"])
    return bpy.data.objects.new(name, mesh)
