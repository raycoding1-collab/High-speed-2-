"""
Hyper-realistic procedural model of the JR Central L0 Series SCMaglev (Improved 950-series)
built to exact metric specifications from `src/specs.py`.

Features:
- Exact 15.0 m Aero-Long-Nose parametric loft (28.0 m end cars, 24.3 m intermediate cars)
- High-resolution (4096x2048) procedurally generated livery, panel seams, wear, low dirt,
  and nose bug strikes baked into packed textures
- Flush high-mounted dark camera visor (950-series) with internal twin lenses & LED lamps
- Recessed passenger porthole windows with tinted reflective glass over full 3D cabin interior
  (2+2 turquoise seats, white headrests, off-white curved walls, warm LED ceiling strip)
- Jacobs articulated superconducting magnet bogies with retracted wheels & visible 10 cm air gap
- Full 12-car (299.0 m) instanced formation
"""
import math
import os
import bpy
import bmesh
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from mathutils import Vector, Matrix

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import specs


# ==============================================================================
# 1. PARAMETRIC L0-950 CROSS-SECTION & NOSE PROFILE FUNCTIONS
# ==============================================================================
def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def l0_nose_envelope(s: float):
    """
    Given distance `s` from the front nose tip (s=0.0 m tip -> s=15.0 m full cabin body):
    Returns (half_width, z_bottom, z_top, roof_dome_power, lower_squareness)
    calibrated directly against JR Central L0-950 elevation & 3/4 reference photos.
    """
    L = specs.NOSE_LENGTH  # 15.0 m
    hw_full = specs.CAR_WIDTH * 0.5   # 1.45 m
    zb_full = specs.BODY_BOTTOM_Z     # 0.42 m
    zt_full = specs.CAR_TOP_Z         # 3.52 m

    if s >= L:
        return hw_full, zb_full, zt_full, 2.2, 4.2

    u = max(0.0, min(1.0, s / L))  # 0 at nose tip, 1 at full cabin

    # Width profile: rounded bullnose tip (0..1.5m), sleek widening cheeks to full width by s=12.5m
    if u < 0.04:
        w_frac = math.sqrt(max(1e-5, u / 0.04)) * 0.52
    else:
        u2 = (u - 0.04) / 0.96
        w_frac = 0.52 + 0.48 * (1.0 - (1.0 - min(1.0, u2 / 0.85)) ** 2.15)
    half_w = hw_full * w_frac

    # Bottom profile: slightly upswept aerodynamic splitter nose tip, dropping to flat floor by s=2.8m
    zb = zb_full + 0.24 * ((1.0 - _smoothstep(0.0, 0.22, u)) ** 1.8)

    # Top profile: iconic L0-950 "aero-long-nose" duck-bill silhouette:
    # - Rounded low nose tip (0..2.2m) rising quickly to ~1.18m
    # - Long low sleek fore-deck (2.2..6.5m) rising gently from 1.18m to 1.85m
    # - Sinuous S-curve windshield/camera ramp (6.5..12.2m) climbing to 3.38m
    # - Smooth crown merge into full 3.52m roof by s=15.0m
    tip_h = 0.78 * (math.sqrt(max(1e-5, u / 0.04)) if u < 0.04 else 1.0)
    if u < 0.04:
        zt = zb + tip_h * 0.82
    else:
        # Blend low duckbill wedge + S-curve canopy ramp
        fore_rise = 0.50 * _smoothstep(0.04, 0.45, u)
        canopy_rise = 1.92 * (_smoothstep(0.32, 0.88, u) ** 1.12)
        crown_blend = 0.14 * _smoothstep(0.75, 1.0, u)
        zt = zb_full + 0.64 + fore_rise + canopy_rise + crown_blend

    # Cross-section superellipse exponent: rounder near nose tip, boxy-rounded on cabin
    roof_pow = 1.85 + 0.45 * _smoothstep(0.25, 0.9, u)
    lower_sq = 2.4 + 2.0 * _smoothstep(0.2, 0.85, u)
    return half_w, zb, zt, roof_pow, lower_sq


def sample_cross_section(s: float, n_ring: int = 48):
    """
    Returns list of (y, z) points around the L0-950 cross-section at station `s` (from nose tip).
    Counter-clockwise starting from top center (y=0, z=zt), going +Y (left), bottom, -Y (right).
    """
    hw, zb, zt, roof_pow, lower_sq = l0_nose_envelope(s)
    h = max(0.08, zt - zb)
    zm = zb + h * 0.46  # shoulder/waistline height
    hu = zt - zm
    hl = zm - zb

    pts = []
    for i in range(n_ring):
        theta = 2.0 * math.pi * i / n_ring  # 0 = top (+Z), pi/2 = +Y (left), pi = bottom (-Z)
        cy = math.sin(theta)
        cz = math.cos(theta)
        if cz >= 0.0:
            # Upper half: smooth domed shoulder with slight inward tumblehome near roof
            p = roof_pow
            sy = math.copysign(abs(cy) ** (2.0 / p), cy)
            sz = (cz ** (2.0 / p))
            # Subtle aerodynamic side-shoulder taper on nose
            y = hw * sy
            z = zm + hu * sz
        else:
            # Lower half: squared-off lower body (RTRI L0 spec for low bogie drag)
            p = lower_sq
            sy = math.copysign(abs(cy) ** (2.0 / p), cy)
            sz = -((-cz) ** (2.0 / p))
            # Slight inward bevel on very bottom edge
            y = hw * sy * (0.96 + 0.04 * (1.0 + sz))
            z = zm + hl * sz
        pts.append((y, z))
    return pts


# ==============================================================================
# 2. HIGH-RESOLUTION PROCEDURAL TEXTURE GENERATION (L0-950 LIVERY + WEAR)
# ==============================================================================
def generate_l0_textures(tex_dir: str) -> dict:
    """
    Generate crisp 4096x2048 albedo + roughness + bump textures for:
    - End car (L21/L22-950): nose converging stripes, camera visor mask, side wave stripe,
      L0 emblem, double upper window stripes, panel seams, low grime, nose bug strikes.
    - Intermediate car (L25-950): double upper window stripes, panel seams, door outlines, low grime.
    UV mapping convention:
      u in [0, 1] = longitudinal s / car_length (0 = nose tip or car front, 1 = car rear)
      v in [0, 1] = circumferential ring parameter:
          v = 0.00 / 1.00 : top centerline (roof crown)
          v = 0.25        : left side (+Y) waistline
          v = 0.50        : bottom centerline
          v = 0.75        : right side (-Y) waistline
    """
    os.makedirs(tex_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    paths = {}
    for kind, car_len in [("end", specs.END_CAR_LENGTH), ("mid", specs.MID_CAR_LENGTH)]:
        alb_path = os.path.join(tex_dir, f"l0_950_{kind}_albedo.png")
        rgh_path = os.path.join(tex_dir, f"l0_950_{kind}_rough.png")
        bmp_path = os.path.join(tex_dir, f"l0_950_{kind}_bump.png")
        paths[kind] = {"albedo": alb_path, "rough": rgh_path, "bump": bmp_path}

        W, H = 4096, 2048
        # Base pearl white (#F2F4F7 — physically realistic bright white albedo ~0.82 linear, never 1.0!)
        alb = np.full((H, W, 3), [236, 239, 243], dtype=np.uint8)
        rgh = np.full((H, W), 24, dtype=np.uint8)   # ~0.095 roughness glossy clearcoat
        bmp = np.full((H, W), 128, dtype=np.uint8)

        # Normalized coordinate grids
        u = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]  # [1, W]
        v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # [H, 1]
        s_m = u * car_len  # meters along car

        # Distance from roof centerline (0 at roof center, 0.25 at side waist, 0.5 at bottom)
        d_roof = np.minimum(v, 1.0 - v)  # [0, 0.5]

        # JR Central Deep Cobalt Blue (#0C2E6E -> RGB 12, 46, 110)
        BLUE = np.array([13, 47, 112], dtype=np.uint8)
        DARK_VISOR = np.array([10, 13, 19], dtype=np.uint8)
        SEAM_GREY = np.array([150, 156, 165], dtype=np.uint8)

        if kind == "end":
            # 1) NOSE ROOF CONVERGING BLUE STRIPES (s = 3.2m .. 13.8m, tapering toward tip)
            # Matches L0-950.jpg & 20220528yamanashi01.jpg: wide center band + 2 thin flanking lines
            nose_u = np.clip((s_m - 2.6) / (13.4 - 2.6), 0.0, 1.0)
            in_nose_stripe = (s_m >= 2.6) & (s_m <= 13.6)
            # Width expands smoothly from sharp spearhead point at s=2.6m to full roof width at s=12.0m
            spread = (nose_u ** 0.72)
            # Center twin-split stripe + outer pinstripe on each side
            c_band = (d_roof < 0.044 * spread) & (d_roof > 0.005 * spread)
            f_band1 = (d_roof > 0.052 * spread) & (d_roof < 0.068 * spread)
            f_band2 = (d_roof > 0.076 * spread) & (d_roof < 0.085 * spread)
            nose_blue = in_nose_stripe & (c_band | f_band1 | f_band2)
            # Soft rounded rear termination of roof stripes at s ~ 13.5m
            rear_taper = (s_m > 12.6) & (d_roof > 0.085 * (1.0 - (s_m - 12.6) / 1.2))
            nose_blue &= ~rear_taper
            alb[nose_blue] = BLUE

            # 2) HIGH-MOUNTED DARK CAMERA VISOR (s = 9.35m .. 11.45m, centered on nose slope)
            vis_u = (s_m - 10.40) / 1.02
            vis_v = d_roof / 0.048
            # Rounded trapezoid/oval visor
            visor_mask = ((vis_u ** 2) + (vis_v ** 2) * (1.0 + 0.25 * vis_u)) < 1.0
            visor_border = (((vis_u ** 2) + (vis_v ** 2) * (1.0 + 0.25 * vis_u)) < 1.14) & ~visor_mask
            alb[visor_border] = np.array([28, 34, 44], dtype=np.uint8)
            alb[visor_mask] = DARK_VISOR
            rgh[visor_mask] = 8  # Glassy optical cover
            bmp[visor_border] = 95

            # 3) DRAMATIC SWEEPING SIDE WAVE STRIPE ON NOSE CHEEKS (both +Y and -Y sides)
            # Starts low-mid at s=6.5m, arches up over L0 emblem at s=13.5m, sweeps back to cabin stripes!
            in_wave = (s_m >= 6.4) & (s_m <= 18.8)
            t_w = np.clip((s_m - 6.4) / (18.8 - 6.4), 0.0, 1.0)
            # Center curve in d_roof space: sweeps from 0.29 (low cheek) up to 0.165 (mid-upper side) then down slightly
            wave_center = 0.295 - 0.125 * np.maximum(0.0, np.sin(np.pi * np.clip(t_w * 1.15, 0.0, 1.0))) ** 1.2
            wave_half_w = 0.015 * np.maximum(0.0, np.sin(np.pi * t_w)) ** 0.65
            wave_mask = in_wave & (np.abs(d_roof - wave_center) < wave_half_w)
            alb[wave_mask] = BLUE

            # Upper twin taper streaks above the wave leading into cabin stripes (s = 14.0 .. 18.5 m)
            in_streak = (s_m >= 14.5) & (s_m <= specs.END_CAR_LENGTH)
            st_t = np.clip((s_m - 14.5) / 3.5, 0.0, 1.0)
            # Double horizontal cabin stripes ABOVE the passenger windows (d_roof ~ 0.128..0.148 & 0.156..0.164)
            s1_mask = in_streak & (np.abs(d_roof - 0.136) < 0.010 * (0.25 + 0.75 * st_t))
            s2_mask = in_streak & (np.abs(d_roof - 0.158) < 0.0045 * (0.20 + 0.80 * st_t))
            alb[s1_mask | s2_mask] = BLUE

            # Nose tip subtle access hatch & tow cover seam (s = 1.6m)
            tip_seam = (np.abs(s_m - 1.55) < 0.018) & (d_roof > 0.18) & (d_roof < 0.42)
            alb[tip_seam] = SEAM_GREY
            bmp[tip_seam] = 90

            # Subtle bug-strike specks on forward nose tip & leading edges (s < 4.8m)
            for _ in range(180):
                bs = rng.uniform(0.05, 4.6)
                bv_side = rng.choice([0.0, 1.0])
                bd = abs(rng.normal(0.0, 0.14))
                bv = (bd if bv_side == 0.0 else 1.0 - bd) % 1.0
                px = int(np.clip((bs / car_len) * W, 2, W - 3))
                py = int(np.clip(bv * H, 2, H - 3))
                rad = int(rng.integers(1, 3))
                col = rng.integers(45, 95)
                alb[py - rad:py + rad + 1, px - rad:px + rad + 1] = [col + 12, col + 6, col - 5]
                rgh[py - rad:py + rad + 1, px - rad:px + rad + 1] = 110
        else:
            # INTERMEDIATE CAR (L25-950): Continuous double horizontal cobalt blue stripes above windows
            s1_mask = (np.abs(d_roof - 0.136) < 0.010)
            s2_mask = (np.abs(d_roof - 0.158) < 0.0045)
            alb[(s1_mask | s2_mask)[:, 0], :] = BLUE

        # 4) PANEL SEAMS & FLUSH PLUG DOOR OUTLINES (both end & intermediate cars)
        # Longitudinal panel seam along lower skirt & roof shoulder
        for dv in [0.095, 0.345]:
            sm = np.broadcast_to(np.abs(d_roof - dv) < 0.0012, (H, W)).copy()
            if kind == "end":
                sm = sm & (s_m > 12.0)
            alb[sm] = (alb[sm].astype(np.float32) * 0.88).astype(np.uint8)
            bmp[sm] = 104

        # Flush plug passenger door outline near rear vestibule of each car
        door_x = car_len - 2.15
        for side_v in [0.25, 0.75]:
            # Door spans d_roof ~ 0.165 .. 0.345 (height ~1.92 m)
            d_top = (side_v - 0.085) if side_v < 0.5 else (side_v + 0.085)
            d_bot = (side_v + 0.095) if side_v < 0.5 else (side_v - 0.095)
            vmin, vmax = min(d_top, d_bot), max(d_top, d_bot)
            in_dv = (v >= vmin) & (v <= vmax)
            in_du = (s_m >= door_x - 0.44) & (s_m <= door_x + 0.44)
            edge_u = ((np.abs(s_m - (door_x - 0.44)) < 0.016) | (np.abs(s_m - (door_x + 0.44)) < 0.016)) & in_dv
            edge_v = ((np.abs(v - vmin) < 0.0016) | (np.abs(v - vmax) < 0.0016)) & in_du
            alb[edge_u | edge_v] = np.array([125, 132, 142], dtype=np.uint8)
            bmp[edge_u | edge_v] = 85

        # 5) REALISTIC LOWER-BODY AERODYNAMIC GRIME / ROAD-TUNNEL DUST (d_roof > 0.33)
        # Subtle multi-frequency noise concentrated near the lower bogie skirt & underbelly
        noise_low = rng.normal(0.0, 1.0, (H // 8, W // 8)).astype(np.float32)
        noise_img = Image.fromarray(((noise_low * 0.18 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        noise_up = np.array(noise_img.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0

        grime_band = np.clip((d_roof - 0.31) / 0.14, 0.0, 1.0) ** 1.6
        grime_amt = grime_band * (0.22 + 0.25 * noise_up)
        # Darker technical grey on the very bottom belly (d_roof > 0.41)
        belly = np.broadcast_to(np.clip((d_roof - 0.41) / 0.04, 0.0, 1.0), (H, W))
        grime_col = np.array([138, 134, 126], dtype=np.float32)
        belly_col = np.array([62, 66, 72], dtype=np.float32)

        alb_f = alb.astype(np.float32)
        alb_f = alb_f * (1.0 - grime_amt[:, :, None]) + grime_col * grime_amt[:, :, None]
        alb_f = alb_f * (1.0 - belly[:, :, None]) + belly_col * belly[:, :, None]
        alb = np.clip(alb_f, 0, 255).astype(np.uint8)

        rgh_f = rgh.astype(np.float32) + grime_amt * 95.0 + belly * 120.0 + (noise_up - 0.5) * 14.0
        rgh = np.clip(rgh_f, 6, 230).astype(np.uint8)

        # Draw subtle stylized "L0" emblem on end car flank (s ~ 13.3m, d_roof ~ 0.245)
        img_alb = Image.fromarray(alb, mode="RGB")
        if kind == "end":
            draw = ImageDraw.Draw(img_alb)
            ex = int((13.3 / car_len) * W)
            for side_idx, ey in enumerate([int(0.245 * H), int(0.755 * H)]):
                # Gold/blue swoosh ring + "L0" geometric mark oriented upright on both sides
                draw.arc([ex - 52, ey - 32, ex + 52, ey + 32], start=20, end=320, fill=(185, 160, 82), width=6)
                if side_idx == 0:
                    draw.line([ex - 26, ey - 18, ex - 26, ey + 18, ex - 4, ey + 18], fill=(13, 47, 112), width=8)
                    draw.ellipse([ex + 4, ey - 18, ex + 32, ey + 18], outline=(13, 47, 112), width=7)
                else:
                    # On -Y side, +v goes DOWN toward floor while on +Y side +v goes DOWN too, wait: u increases toward rear (-X)
                    # Viewing right side (-Y): front is RIGHT (+X), rear is LEFT (-X), so +u goes LEFT! Draw "0" at left, "L" at right mirrored in X:
                    draw.ellipse([ex - 32, ey - 18, ex - 4, ey + 18], outline=(13, 47, 112), width=7)
                    draw.line([ex + 26, ey - 18, ex + 26, ey + 18, ex + 4, ey + 18], fill=(13, 47, 112), width=8)

        img_alb.save(alb_path)
        Image.fromarray(rgh, mode="L").save(rgh_path)
        Image.fromarray(bmp, mode="L").save(bmp_path)

    return paths


# ==============================================================================
# 3. MATERIALS
# ==============================================================================
def _load_img(path: str, colorspace: str = "sRGB"):
    name = os.path.basename(path)
    if name in bpy.data.images:
        img = bpy.data.images[name]
        img.reload()
    else:
        img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    return img


def make_car_body_material(name: str, tex_dict: dict) -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.06  # Subtle aerospace pearl paint
    bsdf.inputs["IOR"].default_value = 1.50
    bsdf.inputs["Coat Weight"].default_value = 0.95
    bsdf.inputs["Coat Roughness"].default_value = 0.035
    bsdf.inputs["Coat IOR"].default_value = 1.55

    tex_alb = nt.nodes.new("ShaderNodeTexImage")
    tex_alb.image = _load_img(tex_dict["albedo"], "sRGB")

    tex_rgh = nt.nodes.new("ShaderNodeTexImage")
    tex_rgh.image = _load_img(tex_dict["rough"], "Non-Color")

    tex_bmp = nt.nodes.new("ShaderNodeTexImage")
    tex_bmp.image = _load_img(tex_dict["bump"], "Non-Color")

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.008

    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex_rgh.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(tex_bmp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_simple_mat(name: str, color, rough=0.35, metallic=0.0, emission=None, em_strength=0.0,
                    transmission=0.0, ior=1.45, alpha=1.0, coat=0.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Coat Weight"].default_value = coat
    if emission is not None and em_strength > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = em_strength
    return mat


# ==============================================================================
# 4. MESH BUILDERS (SHELL, RECESSED WINDOWS, INTERIOR, BOGIES, HEADLIGHTS)
# ==============================================================================
def _build_lofted_shell_mesh(name: str, s_stations: list, car_len: float, n_ring: int = 52,
                             constant_section: bool = False) -> bpy.types.Mesh:
    """
    Build lofted car shell along X in [0, -car_len] where X=0 is nose tip (end car) or front bulkhead (mid car).
    When constant_section=True (intermediate cars), every station uses the full cabin cross-section!
    """
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    rings = []
    for s in s_stations:
        s_prof = (specs.NOSE_LENGTH + 2.0) if constant_section else s
        pts_yz = sample_cross_section(s_prof, n_ring=n_ring)
        x = -s  # Front at x=0, body extends in -X
        ring = [bm.verts.new((x, y, z)) for (y, z) in pts_yz]
        rings.append((s, ring))

    # Front cap
    s0, r0 = rings[0]
    if constant_section:
        bm.faces.new(list(reversed(r0)))
    else:
        hw0, zb0, zt0, _, _ = l0_nose_envelope(s0)
        tip_v = bm.verts.new((0.06, 0.0, 0.5 * (zb0 + zt0) - 0.04))
        for j in range(n_ring):
            jn = (j + 1) % n_ring
            f = bm.faces.new((tip_v, r0[jn], r0[j]))
            f.smooth = True

    # Quad strips between rings
    for idx in range(len(rings) - 1):
        sa, ra = rings[idx]
        sb, rb = rings[idx + 1]
        ua = sa / car_len
        ub = sb / car_len
        for j in range(n_ring):
            jn = (j + 1) % n_ring
            f = bm.faces.new((ra[j], ra[jn], rb[jn], rb[j]))
            f.smooth = True
            va = j / n_ring
            vb = (j + 1) / n_ring
            uvs = [(ua, va), (ua, vb), (ub, vb), (ub, va)]
            for loop, uv in zip(f.loops, uvs):
                loop[uv_layer].uv = uv

    # Rear bulkhead cap
    _, r_last = rings[-1]
    bm.faces.new(r_last)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _build_windows_and_interior(parent_obj, car_len: float, win_s_positions: list,
                                mat_glass, mat_gasket, mat_wall, mat_seat, mat_headrest, mat_led):
    """
    Create flush/slightly-recessed airplane-style vertical rounded-rectangle porthole windows
    on BOTH sides (+Y and -Y) plus visible 3D cabin interior (2+2 seats, white headrests, LED strip)
    directly behind the windows!
    """
    # Because windows sit cleanly on the flat/near-flat vertical side of the L0 cabin (y = +-1.45m),
    # we place a crisp dark rubber bezel + recessed glass pane with transparent cutaway?
    # Wait: to actually see INSIDE through the windows in Cycles without Boolean artifacts on the loft,
    # we model the window pane slightly proud of a recessed dark interior bay OR place real cutout openings?
    # EVEN BETTER: we build the cabin windows as genuine glass portals mounted over a bright illuminated
    # interior box using a crisp shader alpha cutout on the car shell OR actual recessed window geometry!
    # Let's check: real alpha cutout on the shell at exact window rectangles gives 100% clean shading
    # with ZERO Boolean triangulation distortion on curved glossy clearcoat paint!
    # Wait — let's build BOTH: real 3D rubber window bezels + glass panes AND a crisp 3D cabin interior
    # inside the car, AND we cut out the window alpha mask cleanly in the shell material (transparent BSDF
    # mix for window holes) so rays pass straight through the glossy glass into the lit 3D seats & ceiling!
    bm_bezel = bmesh.new()
    bm_glass = bmesh.new()
    bm_int = bmesh.new()
    bm_seat = bmesh.new()
    bm_head = bmesh.new()
    bm_led = bmesh.new()

    hw = specs.CAR_WIDTH * 0.5
    ww = specs.WINDOW_WIDTH
    wh = specs.WINDOW_HEIGHT
    wz = specs.WINDOW_Z_CENTER

    # Build window frames & glass on both sides
    for s in win_s_positions:
        x = -s
        for side in (1.0, -1.0):
            y_out = side * (hw + 0.004)
            y_in = side * (hw - 0.05)
            # Rounded-corner outer rubber bezel (4 frame bars)
            t = 0.028
            for (dx, dz, sx, sz) in [
                (0.0, wh * 0.5, ww + 2 * t, t),
                (0.0, -wh * 0.5, ww + 2 * t, t),
                (-ww * 0.5, 0.0, t, wh),
                (ww * 0.5, 0.0, t, wh),
            ]:
                m = Matrix.Translation((x + dx, side * (hw - 0.015), wz + dz)) @ Matrix.Diagonal((sx, 0.045, sz, 1.0))
                bmesh.ops.create_cube(bm_bezel, size=1.0, matrix=m)
            # Glossy tinted glass pane
            mg = Matrix.Translation((x, side * (hw - 0.012), wz)) @ Matrix.Diagonal((ww, 0.012, wh, 1.0))
            bmesh.ops.create_cube(bm_glass, size=1.0, matrix=mg)

    # 3D Cabin Interior (visible through the cut-out windows!)
    s_min = min(win_s_positions) - 0.8
    s_max = max(win_s_positions) + 0.8
    xc = -0.5 * (s_min + s_max)
    clen = s_max - s_min
    # Floor & curved ceiling shell (strictly inside outer 2.90m shell & rounded roof crown!)
    bmesh.ops.create_cube(bm_int, size=1.0,
                          matrix=Matrix.Translation((xc, 0.0, specs.BODY_BOTTOM_Z + 0.25)) @ Matrix.Diagonal((clen, 2.40, 0.08, 1.0)))
    bmesh.ops.create_cube(bm_int, size=1.0,
                          matrix=Matrix.Translation((xc, 0.0, specs.CAR_TOP_Z - 0.42)) @ Matrix.Diagonal((clen, 1.62, 0.08, 1.0)))
    # Warm-white continuous LED ceiling light strip
    bmesh.ops.create_cube(bm_led, size=1.0,
                          matrix=Matrix.Translation((xc, 0.0, specs.CAR_TOP_Z - 0.48)) @ Matrix.Diagonal((clen, 0.26, 0.04, 1.0)))
    # 2+2 turquoise-blue seats aligned with each window row + white headrest covers
    for s in win_s_positions:
        x = -s - 0.08
        for sy in (-0.95, -0.45, 0.45, 0.95):
            # Seat back (height 1.23m per L0-950 spec)
            bmesh.ops.create_cube(bm_seat, size=1.0,
                                  matrix=Matrix.Translation((x - 0.18, sy, specs.BODY_BOTTOM_Z + 0.92)) @ Matrix.Diagonal((0.14, 0.44, 0.82, 1.0)))
            # Seat cushion
            bmesh.ops.create_cube(bm_seat, size=1.0,
                                  matrix=Matrix.Translation((x + 0.06, sy, specs.BODY_BOTTOM_Z + 0.56)) @ Matrix.Diagonal((0.44, 0.44, 0.14, 1.0)))
            # White antimacassar headrest cover (signature JR Central look!)
            bmesh.ops.create_cube(bm_head, size=1.0,
                                  matrix=Matrix.Translation((x - 0.10, sy, specs.BODY_BOTTOM_Z + 1.20)) @ Matrix.Diagonal((0.04, 0.38, 0.22, 1.0)))

    created = []
    for suffix, bm, mat in [
        ("WinBezel", bm_bezel, mat_gasket),
        ("WinGlass", bm_glass, mat_glass),
        ("IntShell", bm_int, mat_wall),
        ("IntSeats", bm_seat, mat_seat),
        ("IntHeadrests", bm_head, mat_headrest),
        ("IntLED", bm_led, mat_led),
    ]:
        mesh = bpy.data.meshes.new(f"{parent_obj.name}_{suffix}")
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(f"{parent_obj.name}_{suffix}", mesh)
        obj.parent = parent_obj
        created.append(obj)
    return created


def _patch_window_cutouts_in_material(mat: bpy.types.Material, win_s_positions: list, car_len: float):
    """
    Generate a razor-sharp binary window-hole mask in UV space and mix a Transparent BSDF
    into the car shell material so Cycles rays see right through the glass into the 3D cabin!
    """
    W, H = 2048, 1024
    mask = np.full((H, W), 255, dtype=np.uint8)  # 255 = solid paint, 0 = window opening

    # Compute exact v-band for WINDOW_Z_CENTER +- WINDOW_HEIGHT/2 on full cross-section
    pts = sample_cross_section(specs.NOSE_LENGTH + 2.0, n_ring=256)
    zs = np.array([p[1] for p in pts])
    ys = np.array([p[0] for p in pts])
    z_lo = specs.WINDOW_Z_CENTER - specs.WINDOW_HEIGHT * 0.47
    z_hi = specs.WINDOW_Z_CENTER + specs.WINDOW_HEIGHT * 0.47

    left_idx = np.where((ys > 0.5) & (zs >= z_lo) & (zs <= z_hi))[0]
    right_idx = np.where((ys < -0.5) & (zs >= z_lo) & (zs <= z_hi))[0]
    v_bands = [
        (int(left_idx.min() / 256.0 * H), int(left_idx.max() / 256.0 * H)),
        (int(right_idx.min() / 256.0 * H), int(right_idx.max() / 256.0 * H)),
    ]

    for s in win_s_positions:
        u0 = int(max(0, (s - specs.WINDOW_WIDTH * 0.48) / car_len * W))
        u1 = int(min(W - 1, (s + specs.WINDOW_WIDTH * 0.48) / car_len * W))
        for v0, v1 in v_bands:
            mask[v0:v1, u0:u1] = 0

    tex_dir = os.path.dirname(mat.node_tree.nodes.get("Image Texture").image.filepath)
    mask_path = os.path.join(tex_dir, f"{mat.name}_winmask.png")
    Image.fromarray(np.flipud(mask), mode="L").save(mask_path)

    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tex_m = nt.nodes.new("ShaderNodeTexImage")
    tex_m.image = _load_img(mask_path, "Non-Color")

    nt.links.new(tex_m.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def _build_jacobs_bogie_mesh(name: str) -> bpy.types.Mesh:
    """
    Build a superconducting magnet (SCM) Jacobs bogie with cryostat modules on left/right sides,
    central structural frame, and RETRACTED rubber wheel bay doors — bottom sits cleanly at
    Z = LEVITATION_GAP (0.10 m) above the guideway floor!
    """
    bm = bmesh.new()
    L = specs.BOGIE_LENGTH
    W = specs.BOGIE_WIDTH
    z_bot = specs.LEVITATION_GAP  # 0.10 m levitation gap!
    z_mid = z_bot + 0.36

    # Main central bogie structural spine
    bmesh.ops.create_cube(bm, size=1.0,
                          matrix=Matrix.Translation((0.0, 0.0, z_mid + 0.10)) @ Matrix.Diagonal((L * 0.92, W * 0.68, 0.48, 1.0)))

    # Left & right Superconducting Magnet (SCM) cryostat pods (4 poles per side)
    for side in (-1.0, 1.0):
        y_pod = side * (W * 0.5 - 0.16)
        bmesh.ops.create_cube(bm, size=1.0,
                              matrix=Matrix.Translation((0.0, y_pod, z_mid)) @ Matrix.Diagonal((L * 0.88, 0.30, 0.56, 1.0)))
        # Aerodynamic tapered end cones on cryostat pods
        for ex in (-1.0, 1.0):
            bmesh.ops.create_cube(bm, size=1.0,
                                  matrix=Matrix.Translation((ex * (L * 0.46), y_pod, z_mid)) @ Matrix.Diagonal((0.32, 0.24, 0.44, 1.0)))

    # Retracted rubber landing-gear wheel bay doors (closed flush at Z = 0.10m!)
    for wx in (-1.45, 1.45):
        bmesh.ops.create_cube(bm, size=1.0,
                              matrix=Matrix.Translation((wx, 0.0, z_bot + 0.04)) @ Matrix.Diagonal((0.95, 1.40, 0.08, 1.0)))

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _build_headlight_cluster(parent_obj, is_front: bool, mat_lamp, mat_dark):
    """Add glowing LED headlights (front) or red taillights (rear) inside the L0-950 nose visor & lower slit."""
    bm_l = bmesh.new()
    # Twin LED bars integrated at bottom edge of the 950-series high camera visor (s ~ 9.55m)
    hw, zb, zt, _, _ = l0_nose_envelope(9.55)
    for sy in (-0.26, 0.26):
        bmesh.ops.create_cube(bm_l, size=1.0,
                              matrix=Matrix.Translation((-9.55, sy, zt - 0.14)) @ Matrix.Diagonal((0.18, 0.22, 0.07, 1.0)))
    # Also subtle lower nose tip marker lights (s ~ 0.65m)
    _, zb0, zt0, _, _ = l0_nose_envelope(0.65)
    for sy in (-0.34, 0.34):
        bmesh.ops.create_cube(bm_l, size=1.0,
                              matrix=Matrix.Translation((-0.65, sy, zb0 + 0.22)) @ Matrix.Diagonal((0.14, 0.16, 0.05, 1.0)))

    mesh = bpy.data.meshes.new(f"{parent_obj.name}_Lamps")
    bm_l.to_mesh(mesh)
    bm_l.free()
    mesh.materials.append(mat_lamp)
    obj = bpy.data.objects.new(f"{parent_obj.name}_Lamps", mesh)
    obj.parent = parent_obj
    return obj


# ==============================================================================
# 5. MASTER TRAIN BUILDER (12-CAR 299.0 m FORMATION)
# ==============================================================================
def build_l0_train(collection: bpy.types.Collection, tex_dir: str, num_mid_cars: int = None) -> bpy.types.Object:
    """
    Build the complete JR Central L0-950 trainset parented to a root Empty `L0_950_Root`.
    Root origin (0, 0, 0) is at the FRONT NOSE TIP on trough floor datum (Z=0).
    Train extends along -X (so moving root in +X drives the train forward!).
    """
    if num_mid_cars is None:
        num_mid_cars = specs.NUM_INTERMEDIATE_CARS

    tex_paths = generate_l0_textures(tex_dir)

    mat_end = make_car_body_material("L0_950_EndCar_Paint", tex_paths["end"])
    mat_mid = make_car_body_material("L0_950_MidCar_Paint", tex_paths["mid"])
    mat_glass = make_simple_mat("L0_Window_Glass", (0.04, 0.08, 0.12), rough=0.02, metallic=0.0,
                                transmission=0.82, ior=1.52, coat=1.0)
    mat_gasket = make_simple_mat("L0_Window_Gasket", (0.025, 0.027, 0.032), rough=0.55)
    mat_wall = make_simple_mat("L0_Cabin_Wall", (0.76, 0.75, 0.72), rough=0.60)
    mat_seat = make_simple_mat("L0_Cabin_Seat", (0.07, 0.36, 0.56), rough=0.75)
    mat_head = make_simple_mat("L0_Cabin_Headrest", (0.85, 0.85, 0.83), rough=0.72)
    mat_led = make_simple_mat("L0_Cabin_LED", (1.0, 0.94, 0.82), rough=0.2, emission=(1.0, 0.92, 0.78), em_strength=6.5)
    mat_bogie = make_simple_mat("L0_SCM_Bogie", (0.055, 0.058, 0.064), rough=0.45, metallic=0.65)
    mat_gangway = make_simple_mat("L0_Gangway", (0.14, 0.15, 0.17), rough=0.55)
    mat_headlight = make_simple_mat("L0_Headlight_LED", (1.0, 0.98, 0.94), rough=0.05,
                                    emission=(1.0, 0.96, 0.90), em_strength=35.0)
    mat_taillight = make_simple_mat("L0_Taillight_LED", (0.95, 0.04, 0.03), rough=0.05,
                                    emission=(1.0, 0.05, 0.03), em_strength=28.0)

    # Window positions (s from front of car)
    end_wins = [16.5 + i * 1.15 for i in range(specs.END_CAR_WINDOWS)]
    mid_wins = [2.8 + i * 1.18 for i in range(specs.MID_CAR_WINDOWS)]

    # Patch crisp window holes into body paint shaders
    _patch_window_cutouts_in_material(mat_end, end_wins, specs.END_CAR_LENGTH)
    _patch_window_cutouts_in_material(mat_mid, mid_wins, specs.MID_CAR_LENGTH)

    # Dense stations along 15 m nose + cabin for smooth shading
    end_stations = (
        list(np.linspace(0.0, 1.2, 16, endpoint=False))
        + list(np.linspace(1.2, 15.0, 56, endpoint=False))
        + list(np.linspace(15.0, specs.END_CAR_LENGTH, 18))
    )
    mid_stations = list(np.linspace(0.0, specs.MID_CAR_LENGTH, 20))

    mesh_end = _build_lofted_shell_mesh("L0_950_EndCar_Mesh", end_stations, specs.END_CAR_LENGTH, n_ring=56, constant_section=False)
    mesh_end.materials.append(mat_end)
    mesh_mid = _build_lofted_shell_mesh("L0_950_MidCar_Mesh", mid_stations, specs.MID_CAR_LENGTH, n_ring=56, constant_section=True)
    mesh_mid.materials.append(mat_mid)
    mesh_bogie = _build_jacobs_bogie_mesh("L0_950_Bogie_Mesh")
    mesh_bogie.materials.append(mat_bogie)

    # Gangway shroud mesh (flush ring filling the 4 cm seam between cars)
    bm_gw = bmesh.new()
    pts_gw = sample_cross_section(specs.NOSE_LENGTH + 2.0, n_ring=48)
    r_f = [bm_gw.verts.new((0.06, y * 0.988, z)) for (y, z) in pts_gw]
    r_b = [bm_gw.verts.new((-0.06, y * 0.988, z)) for (y, z) in pts_gw]
    for j in range(48):
        jn = (j + 1) % 48
        bm_gw.faces.new((r_f[j], r_f[jn], r_b[jn], r_b[j])).smooth = True
    bmesh.ops.recalc_face_normals(bm_gw, faces=bm_gw.faces)
    mesh_gw = bpy.data.meshes.new("L0_950_Gangway_Mesh")
    bm_gw.to_mesh(mesh_gw)
    bm_gw.free()
    mesh_gw.materials.append(mat_gangway)

    root = bpy.data.objects.new("L0_950_Root", None)
    root.empty_display_type = "ARROWS"
    collection.objects.link(root)

    def _link(obj):
        collection.objects.link(obj)
        return obj

    # 1) Lead End Car (Car 1)
    c_lead = _link(bpy.data.objects.new("L0_Car01_Lead", mesh_end))
    c_lead.parent = root
    c_lead.location = (0.0, 0.0, 0.0)
    for child in _build_windows_and_interior(c_lead, specs.END_CAR_LENGTH, end_wins,
                                             mat_glass, mat_gasket, mat_wall, mat_seat, mat_head, mat_led):
        _link(child)
    _link(_build_headlight_cluster(c_lead, True, mat_headlight, mat_gasket))

    # Lead bogie at s = 6.40 m behind nose tip
    b_lead = _link(bpy.data.objects.new("L0_Bogie_00_Lead", mesh_bogie))
    b_lead.parent = root
    b_lead.location = (-specs.LEAD_BOGIE_OFFSET, 0.0, 0.0)

    # 2) Intermediate Cars (instanced meshes for zero extra RAM!)
    cur_x = -specs.END_CAR_LENGTH
    # Build template child meshes for mid car windows/interior once, then instance across all mid cars!
    mid_template_children = None
    for idx in range(num_mid_cars):
        # Jacobs bogie & gangway at articulation joint
        bg = _link(bpy.data.objects.new(f"L0_Bogie_{idx + 1:02d}", mesh_bogie))
        bg.parent = root
        bg.location = (cur_x, 0.0, 0.0)

        gw = _link(bpy.data.objects.new(f"L0_Gangway_{idx + 1:02d}", mesh_gw))
        gw.parent = root
        gw.location = (cur_x, 0.0, 0.0)

        cur_x -= specs.CAR_GAP
        car_obj = _link(bpy.data.objects.new(f"L0_Car{idx + 2:02d}_Mid", mesh_mid))
        car_obj.parent = root
        car_obj.location = (cur_x, 0.0, 0.0)

        if mid_template_children is None:
            mid_template_children = _build_windows_and_interior(
                car_obj, specs.MID_CAR_LENGTH, mid_wins,
                mat_glass, mat_gasket, mat_wall, mat_seat, mat_head, mat_led
            )
            for ch in mid_template_children:
                _link(ch)
        else:
            for tmpl in mid_template_children:
                inst = bpy.data.objects.new(f"{car_obj.name}_{tmpl.name.split('_')[-1]}", tmpl.data)
                inst.parent = car_obj
                _link(inst)

        cur_x -= specs.MID_CAR_LENGTH

    # Final articulation Jacobs bogie & gangway before rear end car
    bg_last = _link(bpy.data.objects.new(f"L0_Bogie_{num_mid_cars + 1:02d}", mesh_bogie))
    bg_last.parent = root
    bg_last.location = (cur_x, 0.0, 0.0)
    gw_last = _link(bpy.data.objects.new(f"L0_Gangway_{num_mid_cars + 1:02d}", mesh_gw))
    gw_last.parent = root
    gw_last.location = (cur_x, 0.0, 0.0)

    # 3) Rear End Car (mirrored 180 deg around Z so tail nose points -X!)
    cur_x -= specs.CAR_GAP
    tail_tip_x = cur_x - specs.END_CAR_LENGTH
    c_rear = _link(bpy.data.objects.new(f"L0_Car{num_mid_cars + 2:02d}_Rear", mesh_end))
    c_rear.parent = root
    c_rear.location = (tail_tip_x, 0.0, 0.0)
    c_rear.rotation_euler = (0.0, 0.0, math.pi)
    for child in _build_windows_and_interior(c_rear, specs.END_CAR_LENGTH, end_wins,
                                             mat_glass, mat_gasket, mat_wall, mat_seat, mat_head, mat_led):
        _link(child)
    _link(_build_headlight_cluster(c_rear, False, mat_taillight, mat_gasket))

    b_tail = _link(bpy.data.objects.new(f"L0_Bogie_{num_mid_cars + 2:02d}_Tail", mesh_bogie))
    b_tail.parent = root
    b_tail.location = (tail_tip_x + specs.LEAD_BOGIE_OFFSET, 0.0, 0.0)

    return root
