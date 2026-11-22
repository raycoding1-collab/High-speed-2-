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

    # A) High-contrast FRP sidewall figure-8 ground coil panels + recessed concrete bays (4 modules = 3.60 m x 1.12 m)
    coil_alb = os.path.join(tex_dir, "scmaglev_coils_albedo.png")
    coil_bmp = os.path.join(tex_dir, "scmaglev_coils_bump.png")
    W, H = 2048, 640
    # Slightly darker weathered concrete backing behind bright recessed coil plates
    img_a = Image.new("RGB", (W, H), (146, 148, 146))
    img_b = Image.new("L", (W, H), 110)
    da = ImageDraw.Draw(img_a)
    db = ImageDraw.Draw(img_b)

    # Top overhang shadow band & lower metallic busbar/cable raceway
    da.rectangle([0, 0, W, 38], fill=(78, 80, 82))
    db.rectangle([0, 0, W, 38], fill=55)

    n_mod = 4
    mw = W // n_mod
    for i in range(n_mod):
        x0 = i * mw
        # Bright precast FRP coil carrier plate set inside dark concrete bay seam
        da.rectangle([x0 + 14, 42, x0 + mw - 14, H - 72], fill=(208, 211, 210))
        db.rectangle([x0 + 14, 42, x0 + mw - 14, H - 72], fill=168)
        # Deep vertical precast expansion joint between coil bays
        da.rectangle([x0, 0, x0 + 14, H], fill=(64, 68, 72))
        db.rectangle([x0, 0, x0 + 14, H], fill=35)
        # Upper and lower figure-8 rounded-rectangle propulsion/levitation coil recesses
        for cy in (int(H * 0.29), int(H * 0.67)):
            rx0, rx1 = x0 + 44, x0 + mw - 44
            ry0, ry1 = cy - 96, cy + 96
            # Recessed dark shadow channel + raised inner coil loop
            da.rounded_rectangle([rx0, ry0, rx1, ry1], radius=44, fill=(184, 187, 188), outline=(92, 96, 102), width=14)
            db.rounded_rectangle([rx0, ry0, rx1, ry1], radius=44, fill=142, outline=72, width=16)
            da.rounded_rectangle([rx0 + 28, ry0 + 26, rx1 - 28, ry1 - 26], radius=24, outline=(125, 130, 134), width=8)
            db.rounded_rectangle([rx0 + 28, ry0 + 26, rx1 - 28, ry1 - 26], radius=24, outline=190, width=8)
            # Dark mounting bolt recesses (unmistakable repeating dot array in Yamanashi photos!)
            for bx in (x0 + 78, x0 + mw // 2, x0 + mw - 78):
                for by in (cy - 44, cy + 44):
                    da.ellipse([bx - 12, by - 12, bx + 12, by + 12], fill=(38, 42, 46))
                    db.ellipse([bx - 13, by - 13, bx + 13, by + 13], fill=40)
        # Lower horizontal dark power/cable raceway & drainage channel
        da.rectangle([x0, H - 72, x0 + mw, H - 44], fill=(82, 86, 90))
        db.rectangle([x0, H - 72, x0 + mw, H - 44], fill=75)
        da.rectangle([x0, H - 44, x0 + mw, H], fill=(112, 110, 104))

    # Subtle grime & rain variation
    arr_a = np.array(img_a, dtype=np.float32)
    noise = rng.normal(0, 6.0, (H, W, 1)).astype(np.float32)
    streak = rng.normal(0, 8.0, (1, W, 1)).astype(np.float32)
    arr_a = np.clip(arr_a + noise + streak * np.linspace(1.0, 0.25, H)[:, None, None], 0, 255).astype(np.uint8)
    Image.fromarray(arr_a).save(coil_alb)
    img_b.save(coil_bmp)

    # B) Weathered Japanese Viaduct Concrete (formwork joints, tie-rod holes, rainwater streaks, efflorescence)
    conc_alb = os.path.join(tex_dir, "viaduct_concrete_albedo.png")
    conc_rgh = os.path.join(tex_dir, "viaduct_concrete_rough.png")
    conc_bmp = os.path.join(tex_dir, "viaduct_concrete_bump.png")
    CW, CH = 2048, 1024
    base = np.full((CH, CW, 3), [148, 146, 140], dtype=np.float32)

    for scale, amp in [(32, 12.0), (128, 7.0), (512, 4.5)]:
        n = rng.normal(0.0, amp, (max(4, CH // scale), max(4, CW // scale))).astype(np.float32)
        ni = Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8)).resize((CW, CH), Image.BILINEAR)
        base += (np.array(ni, dtype=np.float32) - 128.0)[:, :, None]

    bmp_c = np.full((CH, CW), 135, dtype=np.float32)
    for jx in range(0, CW, 256):
        shift = rng.uniform(-7.0, 7.0)
        base[:, jx:min(CW, jx + 256), :] += shift
        base[:, jx:min(CW, jx + 5), :] -= 38.0
        bmp_c[:, jx:min(CW, jx + 5)] = 70.0
        drip_h = int(rng.uniform(CH * 0.35, CH * 0.85))
        grad = np.linspace(1.0, 0.0, drip_h)[:, None, None]
        base[:drip_h, max(0, jx - 14):min(CW, jx + 18), :] -= 22.0 * grad

    streaks = rng.normal(0, 1.0, (1, CW // 4)).astype(np.float32)
    si = np.array(Image.fromarray(np.clip(streaks * 40 + 128, 0, 255).astype(np.uint8)).resize((CW, CH), Image.BILINEAR), dtype=np.float32) - 128.0
    v_fade = np.linspace(1.0, 0.25, CH)[:, None]
    base -= np.clip(si * v_fade * 0.35, -10.0, 22.0)[:, :, None]

    for ty in (int(CH * 0.28), int(CH * 0.68)):
        for tx in range(64, CW, 128):
            base[ty - 4:ty + 5, tx - 4:tx + 5] -= 45.0
            bmp_c[ty - 4:ty + 5, tx - 4:tx + 5] = 65.0

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


def _add_concrete_seams_and_streaks(mat: bpy.types.Material):
    """
    Inject world-space precast concrete panel seams (3.2 m x 2.0 m formwork blocks with subtle
    block-to-block tonal shift) AND vertical rainwater streaking under concrete overhangs into
    a box-projected PBR concrete material so viaduct girders, piers & portals never read smooth!
    """
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    base_link = bsdf.inputs["Base Color"].links[0]
    src_col = base_link.from_socket
    nt.links.remove(base_link)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep_nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep_nrm.inputs["Vector"])

    # 1) Precast concrete panel seams + subtle panel-to-panel color/value shift (using world XY+Z brick)
    sep_pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])
    add_xy = nt.nodes.new("ShaderNodeMath")
    add_xy.operation = "ADD"
    nt.links.new(sep_pos.outputs["X"], add_xy.inputs[0])
    nt.links.new(sep_pos.outputs["Y"], add_xy.inputs[1])
    comb_uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(add_xy.outputs["Value"], comb_uv.inputs["X"])
    nt.links.new(sep_pos.outputs["Z"], comb_uv.inputs["Y"])

    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 0.32
    brick.inputs["Brick Width"].default_value = 1.05
    brick.inputs["Row Height"].default_value = 0.72
    brick.inputs["Mortar Size"].default_value = 0.016
    brick.inputs["Mortar Smooth"].default_value = 0.35
    brick.inputs["Color1"].default_value = (0.93, 0.92, 0.90, 1.0)
    brick.inputs["Color2"].default_value = (1.06, 1.05, 1.03, 1.0)
    brick.inputs["Mortar"].default_value = (0.48, 0.47, 0.45, 1.0)
    nt.links.new(comb_uv.outputs["Vector"], brick.inputs["Vector"])

    mix_seam = nt.nodes.new("ShaderNodeMix")
    mix_seam.data_type = "RGBA"
    mix_seam.blend_type = "MULTIPLY"
    mix_seam.inputs["Factor"].default_value = 0.65
    nt.links.new(src_col, mix_seam.inputs[6])
    nt.links.new(brick.outputs["Color"], mix_seam.inputs[7])

    # 2) Vertical rainwater streaking on vertical walls & under overhangs (1 - |Normal.Z|)
    abs_nz = nt.nodes.new("ShaderNodeMath")
    abs_nz.operation = "ABSOLUTE"
    nt.links.new(sep_nrm.outputs["Z"], abs_nz.inputs[0])
    wall_mask = nt.nodes.new("ShaderNodeMath")
    wall_mask.operation = "SUBTRACT"
    wall_mask.use_clamp = True
    wall_mask.inputs[0].default_value = 1.0
    nt.links.new(abs_nz.outputs["Value"], wall_mask.inputs[1])

    mp_str = nt.nodes.new("ShaderNodeMapping")
    mp_str.inputs["Scale"].default_value = (0.55, 0.55, 0.055)
    nt.links.new(geom.outputs["Position"], mp_str.inputs["Vector"])
    n_str = nt.nodes.new("ShaderNodeTexNoise")
    n_str.inputs["Scale"].default_value = 1.0
    n_str.inputs["Detail"].default_value = 6.0
    n_str.inputs["Roughness"].default_value = 0.68
    nt.links.new(mp_str.outputs["Vector"], n_str.inputs["Vector"])

    mr_str = nt.nodes.new("ShaderNodeMapRange")
    mr_str.inputs["From Min"].default_value = 0.42
    mr_str.inputs["From Max"].default_value = 0.72
    mr_str.inputs["To Min"].default_value = 1.0
    mr_str.inputs["To Max"].default_value = 0.62
    nt.links.new(n_str.outputs["Fac"], mr_str.inputs["Value"])

    mul_wm = nt.nodes.new("ShaderNodeMath")
    mul_wm.operation = "MULTIPLY"
    mul_wm.inputs[1].default_value = 0.78
    nt.links.new(wall_mask.outputs["Value"], mul_wm.inputs[0])

    mix_str = nt.nodes.new("ShaderNodeMix")
    mix_str.data_type = "RGBA"
    mix_str.blend_type = "MULTIPLY"
    nt.links.new(mul_wm.outputs["Value"], mix_str.inputs["Factor"])
    nt.links.new(mix_seam.outputs[2], mix_str.inputs[6])
    nt.links.new(mr_str.outputs["Result"], mix_str.inputs[7])
    nt.links.new(mix_str.outputs[2], bsdf.inputs["Base Color"])

    # Feed subtle seam groove into existing Bump normal chain!
    if bsdf.inputs["Normal"].links:
        old_nrm_sock = bsdf.inputs["Normal"].links[0].from_socket
        nt.links.remove(bsdf.inputs["Normal"].links[0])
        bmp_s = nt.nodes.new("ShaderNodeBump")
        bmp_s.inputs["Strength"].default_value = 0.42
        bmp_s.inputs["Distance"].default_value = 0.035
        nt.links.new(brick.outputs["Fac"], bmp_s.inputs["Height"])
        nt.links.new(old_nrm_sock, bmp_s.inputs["Normal"])
        nt.links.new(bmp_s.outputs["Normal"], bsdf.inputs["Normal"])


def make_guideway_materials(tex_dir: str) -> dict:
    from assets.pbr_materials import make_box_pbr_material
    t = generate_guideway_textures(tex_dir)
    mats = {}

    # 1) Real photogrammetry CC0 Poly Haven weathered concrete + precast panel seams & rainwater streaking!
    is_new_conc = "Mat_Viaduct_Concrete" not in bpy.data.materials
    mats["concrete"] = make_box_pbr_material(
        "Mat_Viaduct_Concrete", "concrete", tile_scale=0.18,
        tint_rgb=(0.91, 0.89, 0.85), tint_fac=0.38, normal_strength=1.15,
        rough_mult=1.08, ao_darkening=0.58, ao_dist=2.8, macro_var=0.26
    )
    if is_new_conc:
        _add_concrete_seams_and_streaks(mats["concrete"])

    # 2) Sidewall figure-8 coil panels (crisp relief + subtle specular)
    if "Mat_SCMaglev_Coils" in bpy.data.materials:
        mats["coils"] = bpy.data.materials["Mat_SCMaglev_Coils"]
    else:
        m = bpy.data.materials.new("Mat_SCMaglev_Coils")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.44
        ta = nt.nodes.new("ShaderNodeTexImage")
        ta.image = _load_img(t["coil_albedo"], "sRGB")
        tb = nt.nodes.new("ShaderNodeTexImage")
        tb.image = _load_img(t["coil_bump"], "Non-Color")
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = 0.72
        bmp.inputs["Distance"].default_value = 0.045
        nt.links.new(ta.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tb.outputs["Color"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        mats["coils"] = m

    # 3) Trough floor concrete (real photogrammetry concrete + expansion seams)
    is_new_fl = "Mat_Trough_Floor" not in bpy.data.materials
    mats["floor"] = make_box_pbr_material(
        "Mat_Trough_Floor", "concrete", tile_scale=0.25,
        tint_rgb=(0.68, 0.67, 0.64), tint_fac=0.58, normal_strength=0.95,
        rough_mult=1.12, ao_darkening=0.50, ao_dist=1.6, macro_var=0.22
    )
    if is_new_fl:
        _add_concrete_seams_and_streaks(mats["floor"])

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

        # Left & right structural concrete sidewalls with recessed coil alcove & top coping overhang!
        for sy0, sy1 in [(ty - hw_tr - sw_t, ty - hw_tr), (ty + hw_tr, ty + hw_tr + sw_t)]:
            sw_prof = [(sy0, -0.05), (sy1, -0.05), (sy1, sw_h), (sy0, sw_h)]
            _swept_profile(bm, uv, sw_prof, s_vals, loc, u_scale=0.08, v_scale=0.3, mat_idx=0, closed=True)
        # Inward overhanging top concrete coping lip (0.14 m recess) & lower cable/drainage curb
        for side_s, y_in in [(1.0, ty - hw_tr), (-1.0, ty + hw_tr)]:
            lip_prof = [
                (y_in, sw_h - 0.14), (y_in + side_s * 0.14, sw_h - 0.14),
                (y_in + side_s * 0.14, sw_h + 0.04), (y_in, sw_h + 0.04)
            ]
            _swept_profile(bm, uv, lip_prof, s_vals, loc, u_scale=0.08, v_scale=0.4, mat_idx=0, closed=True)
            curb_prof = [
                (y_in, 0.015), (y_in + side_s * 0.16, 0.015),
                (y_in + side_s * 0.16, 0.24), (y_in, 0.24)
            ]
            _swept_profile(bm, uv, curb_prof, s_vals, loc, u_scale=0.10, v_scale=0.4, mat_idx=2, closed=True)

        # Inner recessed FRP figure-8 ground coil panels lining BOTH inner faces of each U-trough!
        cz0 = specs.COIL_PANEL_Z_BOTTOM
        cz1 = cz0 + specs.COIL_PANEL_HEIGHT
        u_coil = 1.0 / (4.0 * specs.COIL_PANEL_PITCH)
        v_coil = 1.0 / specs.COIL_PANEL_HEIGHT
        # Right inner face (facing +Y), slightly recessed behind the top coping overhang
        c_prof_r = [(ty - hw_tr + 0.05, cz0), (ty - hw_tr + 0.05, cz1)]
        _swept_profile(bm, uv, c_prof_r, s_vals, loc, u_scale=u_coil, v_scale=v_coil, mat_idx=1, closed=False)
        # Left inner face (facing -Y)
        c_prof_l = [(ty + hw_tr - 0.05, cz1), (ty + hw_tr - 0.05, cz0)]
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

    # 2) Arched Double-Column Concrete Piers + Steel Pier-Cap Inspection Walkways (NO ghostly vertical parapet sticks!)
    bm_pier = bmesh.new()
    p_top = -specs.VIADUCT_GIRDER_DEPTH
    p_bot = -pier_height - 45.0
    p_h = p_top - p_bot
    bmesh.ops.create_cube(bm_pier, size=1.0,
                          matrix=Matrix.Translation((0.0, yc, p_top - 1.1)) @ Matrix.Diagonal((2.8, 9.6, 2.2, 1.0)))
    for dy in (-3.1, 3.1):
        bmesh.ops.create_cube(bm_pier, size=1.0,
                              matrix=Matrix.Translation((0.0, yc + dy, p_top - 1.8 - p_h * 0.5)) @ Matrix.Diagonal((2.3, 2.2, p_h, 1.0)))
        sign_y = -1.0 if dy > 0 else 1.0
        haunch_m = (
            Matrix.Translation((0.0, yc + dy + sign_y * 1.45, p_top - 2.65))
            @ Matrix.Rotation(sign_y * math.radians(32.0), 4, "X")
            @ Matrix.Diagonal((2.2, 1.9, 1.4, 1.0))
        )
        bmesh.ops.create_cube(bm_pier, size=1.0, matrix=haunch_m)
    if pier_height > 20.0:
        bmesh.ops.create_cube(bm_pier, size=1.0,
                              matrix=Matrix.Translation((0.0, yc, p_top - pier_height * 0.52)) @ Matrix.Diagonal((1.6, 6.2, 1.6, 1.0)))
    bmesh.ops.bevel(bm_pier, geom=bm_pier.edges[:], offset=0.07, segments=2, affect="EDGES", clamp_overlap=True)
    mesh_pier = bpy.data.meshes.new(f"ViaductPier_{loc.loc_id}")
    bm_pier.to_mesh(mesh_pier)
    bm_pier.free()
    mesh_pier.materials.append(mats["concrete"])

    # Low-profile steel pier-cap inspection platform & flush concrete parapet expansion brackets
    # (Eliminates tall thin vertical poles that smeared into ghostly semi-transparent bars at 500-603 km/h!)
    bm_mast = bmesh.new()
    bmesh.ops.create_cube(bm_mast, size=1.0,
                          matrix=Matrix.Translation((0.0, yc, p_top - 1.9)) @ Matrix.Diagonal((4.0, 10.6, 0.14, 1.0)))
    for py in (yR + 0.21, yL - 0.21):
        bmesh.ops.create_cube(bm_mast, size=1.0,
                              matrix=Matrix.Translation((0.0, py, ph + 0.04)) @ Matrix.Diagonal((0.45, 0.46, 0.08, 1.0)))
    mesh_mast = bpy.data.meshes.new(f"ViaductMasts_{loc.loc_id}")
    bm_mast.to_mesh(mesh_mast)
    bm_mast.free()
    mesh_mast.materials.append(mats["steel"])

    created = [root, obj_gw]
    s_p = math.ceil(s_start / specs.VIADUCT_PIER_SPACING) * specs.VIADUCT_PIER_SPACING
    p_idx = 0
    while s_p <= s_end:
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
            pier_obj.rotation_euler = (0.0, 0.0, eul[2])
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
    Build an authentic Japanese SCMaglev 56 m slotted micro-pressure-wave tunnel buffer hood
    with chamfered portal ribs, deep recessed side & roof pressure-relief vent slots, dark interior
    shadow baffles, and sloped hillside-hugging concrete retaining buttresses (zero boxy slabs!).
    """
    bm = bmesh.new()
    yc = specs.TRACK_SPACING * 0.5
    W = specs.BUFFER_HOOD_WIDTH
    H = specs.BUFFER_HOOD_HEIGHT
    L = specs.BUFFER_HOOD_LENGTH
    wt = 0.80

    dir_sign = -1.0 if not flip else 1.0  # Direction from mouth INTO the mountain

    # 1) Beveled structural concrete mouth collar + recessed dark pressure-relief louver slits
    n_slits = specs.BUFFER_HOOD_SLITS
    seg_len = L / (n_slits * 2 + 1)
    for k in range(n_slits * 2 + 1):
        x0 = dir_sign * (k * seg_len)
        x1 = dir_sign * ((k + 1) * seg_len)
        xc = 0.5 * (x0 + x1)
        is_slit = (k % 2 == 1)

        # Lower solid concrete base wall extending deep below deck to anchor into hillside
        for sy in (yc - W * 0.5, yc + W * 0.5):
            h_wall = H * 0.42 if is_slit else H
            bmesh.ops.create_cube(
                bm, size=1.0,
                matrix=Matrix.Translation((xc, sy, h_wall * 0.5 - 1.2)) @ Matrix.Diagonal((seg_len + 0.08, wt, h_wall + 2.4, 1.0))
            )
            if is_slit:
                # Recessed top lintel beam + angled exterior pressure-relief vent baffle louvres!
                bmesh.ops.create_cube(
                    bm, size=1.0,
                    matrix=Matrix.Translation((xc, sy, H - 0.55)) @ Matrix.Diagonal((seg_len + 0.08, wt, 1.1, 1.0))
                )
                for lv_i in range(3):
                    lx = x0 + dir_sign * ((lv_i + 0.5) / 3.0) * seg_len
                    bmesh.ops.create_cube(
                        bm, size=1.0,
                        matrix=Matrix.Translation((lx, sy, H * 0.65))
                        @ Matrix.Rotation(dir_sign * math.radians(16.0), 4, "Z")
                        @ Matrix.Diagonal((0.28, wt * 0.95, H * 0.46, 1.0))
                    )
        if not is_slit:
            # Heavy beveled exterior structural concrete arch rib (authentic Yamanashi ribbed buffer hood!)
            bmesh.ops.create_cube(
                bm, size=1.0,
                matrix=Matrix.Translation((xc, yc, H - wt * 0.45)) @ Matrix.Diagonal((seg_len + 0.10, W + wt * 1.45, wt * 1.15, 1.0))
            )
            # Chamfered upper cornice shoulders along outer hood edges
            for side in (-1.0, 1.0):
                bmesh.ops.create_cube(
                    bm, size=1.0,
                    matrix=Matrix.Translation((xc, yc + side * (W * 0.5 + 0.12), H - 0.9))
                    @ Matrix.Rotation(side * math.radians(38.0), 4, "X")
                    @ Matrix.Diagonal((seg_len + 0.08, 1.5, 0.85, 1.0))
                )
        else:
            # Roof pressure-relief slot flanked by deep concrete longitudinal box beams
            for sy in (yc - W * 0.36, yc + W * 0.36):
                bmesh.ops.create_cube(
                    bm, size=1.0,
                    matrix=Matrix.Translation((xc, sy, H - wt * 0.5)) @ Matrix.Diagonal((seg_len + 0.08, 1.9, wt, 1.0))
                )

    # 2) Plumb foundation apron directly beneath the buffer hood sidewalls (zero overhanging diagonal fins!)
    for sy in (yc - W * 0.5, yc + W * 0.5):
        bmesh.ops.create_cube(
            bm, size=1.0,
            matrix=Matrix.Translation((dir_sign * L * 0.65, sy, -7.5)) @ Matrix.Diagonal((L * 0.72, wt * 1.15, 15.0, 1.0))
        )

    # 3) Deep dark interior tunnel bore behind buffer hood (300 m long + back light-blocker)
    bore_x = dir_sign * (L + 150.0)
    for sy in (yc - W * 0.48, yc + W * 0.48):
        bmesh.ops.create_cube(bm, size=1.0,
                              matrix=Matrix.Translation((bore_x, sy, H * 0.5)) @ Matrix.Diagonal((300.0, 0.8, H, 1.0)))
    bmesh.ops.create_cube(bm, size=1.0,
                          matrix=Matrix.Translation((bore_x, yc, H)) @ Matrix.Diagonal((300.0, W, 0.8, 1.0)))
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
