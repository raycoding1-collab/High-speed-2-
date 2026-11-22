"""
Photorealistic rural Japanese architecture using scanned CC0 Poly Haven PBR textures
(`grey_roof_tiles_02` kawara tiles, `dark_wooden_planks` aged cedar, `brushed_concrete_03`
weathered plaster & stone) with overhanging eaves, timber beams, window mullions,
traditional Shinto Myojin Torii gate & stone lanterns, rural Japanese utility poles, and car.
"""
import math
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler
from assets.pbr_materials import make_box_pbr_material


def _mat(name: str, rgb, rough=0.65, metallic=0.0, emission=None, em_s=0.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emission is not None and em_s > 0:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = em_s
    return m


def _box(bm, center, size, mat_idx=0, rot_eul=(0.0, 0.0, 0.0)):
    m = Matrix.Translation(center) @ Euler(rot_eul).to_matrix().to_4x4() @ Matrix.Diagonal((*size, 1.0))
    ret = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    for v in ret["verts"]:
        for f in v.link_faces:
            f.material_index = mat_idx


def _gable_roof(bm, width_x, depth_y, z_eave, ridge_h, overhang=0.75, mat_roof=1, mat_ridge=2, mat_timber=5):
    hx = width_x * 0.5 + overhang
    hy = depth_y * 0.5 + overhang
    slope_len = math.hypot(hy, ridge_h)
    ang = math.atan2(ridge_h, hy)
    for side in (-1.0, 1.0):
        cy = side * hy * 0.5
        cz = z_eave + ridge_h * 0.5
        _box(bm, (0.0, cy, cz), (hx * 2.0, slope_len + 0.18, 0.24), mat_idx=mat_roof, rot_eul=(-side * ang, 0.0, 0.0))
        # Dark wooden fascia & eave beam under roof edge
        _box(bm, (0.0, side * (hy - 0.15), z_eave - 0.08), (hx * 2.02, 0.14, 0.18), mat_idx=mat_timber)
    # Prominent curved Japanese kawara ridge cap beam (mune-gawara) + end tiles (onigawara)
    _box(bm, (0.0, 0.0, z_eave + ridge_h + 0.16), (hx * 2.06, 0.36, 0.28), mat_idx=mat_ridge)
    for sx in (-hx, hx):
        _box(bm, (sx, 0.0, z_eave + ridge_h + 0.26), (0.16, 0.44, 0.44), mat_idx=mat_roof)
    # Gable triangle end walls
    for sx in (-width_x * 0.5, width_x * 0.5):
        v0 = bm.verts.new((sx, -depth_y * 0.5, z_eave))
        v1 = bm.verts.new((sx, depth_y * 0.5, z_eave))
        v2 = bm.verts.new((sx, 0.0, z_eave + ridge_h))
        f = bm.faces.new((v0, v1, v2) if sx > 0 else (v1, v0, v2))
        f.material_index = 0


def build_building_library() -> dict:
    # Real photogrammetry CC0 Poly Haven PBR architectural materials with per-instance color/tone variety!
    mat_plaster = make_box_pbr_material(
        "JP_Wall_Plaster_PBR", "concrete", tile_scale=0.35,
        tint_rgb=(0.98, 0.94, 0.85), tint_fac=0.55, normal_strength=0.55,
        rough_mult=1.15, ao_darkening=0.55, ao_dist=1.4, macro_var=0.16, obj_var=0.18
    )
    mat_stucco_w = make_box_pbr_material(
        "JP_Wall_WhiteKura_PBR", "concrete", tile_scale=0.38,
        tint_rgb=(0.94, 0.93, 0.90), tint_fac=0.45, normal_strength=0.45,
        rough_mult=1.12, ao_darkening=0.50, ao_dist=1.4, macro_var=0.14, obj_var=0.12
    )
    mat_cedar = make_box_pbr_material(
        "JP_Wall_Cedar_PBR", "cedar_wood", tile_scale=0.55,
        tint_rgb=(0.55, 0.44, 0.36), tint_fac=0.45, normal_strength=0.95,
        rough_mult=1.05, ao_darkening=0.50, ao_dist=1.2, macro_var=0.18, obj_var=0.22
    )
    mat_kawara = make_box_pbr_material(
        "JP_Roof_Kawara_PBR", "kawara_roof", tile_scale=0.65,
        tint_rgb=(0.35, 0.38, 0.44), tint_fac=0.65, normal_strength=1.2,
        rough_mult=0.72, ao_darkening=0.45, ao_dist=0.9, metallic=0.08, macro_var=0.14, obj_var=0.22
    )
    mat_kawara_russet = make_box_pbr_material(
        "JP_Roof_Russet_PBR", "kawara_roof", tile_scale=0.65,
        tint_rgb=(0.54, 0.34, 0.24), tint_fac=0.70, normal_strength=1.15,
        rough_mult=0.78, ao_darkening=0.45, ao_dist=0.9, metallic=0.06, macro_var=0.16, obj_var=0.20
    )
    mat_kawara_dark = make_box_pbr_material(
        "JP_Roof_Charcoal_PBR", "kawara_roof", tile_scale=0.65,
        tint_rgb=(0.22, 0.24, 0.27), tint_fac=0.72, normal_strength=1.2,
        rough_mult=0.68, ao_darkening=0.45, ao_dist=0.9, metallic=0.12, macro_var=0.14, obj_var=0.18
    )
    mat_ridge = make_box_pbr_material(
        "JP_Roof_Ridge_PBR", "concrete", tile_scale=0.5,
        tint_rgb=(0.82, 0.80, 0.76), tint_fac=0.4, normal_strength=0.6, ao_darkening=0.4, ao_dist=0.6
    )
    mat_found = make_box_pbr_material(
        "JP_Foundation_PBR", "concrete", tile_scale=0.35,
        tint_rgb=(0.55, 0.54, 0.51), tint_fac=0.6, normal_strength=0.9, ao_darkening=0.6, ao_dist=1.4
    )
    mat_win = _mat("JP_House_Window", (0.04, 0.055, 0.07), rough=0.05, emission=(1.0, 0.74, 0.38), em_s=0.55)
    mat_vermilion = _mat("JP_Torii_Vermilion", (0.58, 0.052, 0.018), rough=0.42)
    mat_torii_cap = _mat("JP_Torii_BlackCap", (0.032, 0.032, 0.036), rough=0.45)
    mat_stone = make_box_pbr_material(
        "JP_Shrine_Stone_PBR", "cliff", tile_scale=0.6,
        tint_rgb=(0.75, 0.74, 0.70), tint_fac=0.5, normal_strength=1.0, ao_darkening=0.55, ao_dist=1.0
    )
    mat_car_w = _mat("JP_Car_Silver", (0.58, 0.61, 0.65), rough=0.12, metallic=0.55)
    mat_car_g = _mat("JP_Car_Glass", (0.03, 0.04, 0.05), rough=0.03)
    mat_steel = _mat("JP_Pole_Steel", (0.28, 0.30, 0.32), rough=0.45, metallic=0.80)

    lib = {}

    # 1. Six Distinct Rural Japanese Building Architectures (zero cookie-cutter repetition!)
    # (wx, dy, h1, h2, roof_h, wall_m, roof_m, has_l_wing)
    house_specs = [
        (10.5, 7.6, 3.0, 2.7, 2.35, mat_plaster,  mat_kawara,        False),  # 0: Classic 2F Nagano farmhouse
        (9.4,  6.8, 3.1, 0.0, 2.20, mat_cedar,    mat_kawara_dark,   False),  # 1: 1F charred-cedar bungalow
        (11.8, 8.0, 3.0, 2.5, 2.45, mat_plaster,  mat_kawara_russet, True),   # 2: Large L-wing orchard farmhouse
        (7.2,  5.6, 4.8, 0.0, 2.10, mat_stucco_w, mat_kawara_dark,   False),  # 3: Traditional white kura storehouse
        (13.2, 6.2, 3.6, 0.0, 2.15, mat_cedar,    mat_kawara_russet, False),  # 4: Wide timber agricultural barn
        (8.4,  7.8, 2.9, 2.8, 2.05, mat_stucco_w, mat_kawara,        True),   # 5: Compact 2F L-shaped residence
    ]
    for vi, (wx, dy, h1, h2, roof_h, wall_m, roof_m, has_wing) in enumerate(house_specs):
        bm = bmesh.new()
        _box(bm, (0.0, 0.0, -1.2), (wx + 0.25, dy + 0.25, 3.2), mat_idx=3)
        _box(bm, (0.0, 0.0, 0.4 + h1 * 0.5), (wx, dy, h1), mat_idx=0)
        for sx in (-wx * 0.5, wx * 0.5):
            for sy in (-dy * 0.5, dy * 0.5):
                _box(bm, (sx, sy, 0.4 + h1 * 0.5), (0.24, 0.24, h1), mat_idx=5)
        for side in (-1.0, 1.0):
            win_xs = (-wx * 0.28, 0.0, wx * 0.28) if wx > 9.0 else (-wx * 0.22, wx * 0.22)
            for kx in win_xs:
                _box(bm, (kx, side * (dy * 0.5 + 0.04), 0.4 + h1 * 0.52), (1.50, 0.08, 1.32), mat_idx=5)
                _box(bm, (kx, side * (dy * 0.5 + 0.06), 0.4 + h1 * 0.52), (1.32, 0.06, 1.16), mat_idx=4)
                _box(bm, (kx, side * (dy * 0.5 + 0.08), 0.4 + h1 * 0.52), (0.06, 0.06, 1.18), mat_idx=5)
        if has_wing:
            # Attached 1F side engawa / annex wing breaking up rectangular silhouette
            _box(bm, (wx * 0.36, -dy * 0.62, 0.4 + 1.4), (wx * 0.48, dy * 0.52, 2.8), mat_idx=5)
            _box(bm, (wx * 0.36, -dy * 0.62, 0.4 + 2.95), (wx * 0.54, dy * 0.62, 0.28), mat_idx=1, rot_eul=(0.22, 0.0, 0.0))
        if h2 > 0.1:
            _box(bm, (0.0, 0.0, 0.4 + h1), (wx + 1.25, dy + 1.25, 0.20), mat_idx=1)
            _box(bm, (0.0, 0.0, 0.4 + h1 + h2 * 0.5), (wx * 0.88, dy * 0.88, h2), mat_idx=0)
            for side in (-1.0, 1.0):
                for kx in (-wx * 0.22, wx * 0.22):
                    _box(bm, (kx, side * (dy * 0.44 + 0.04), 0.4 + h1 + h2 * 0.52), (1.45, 0.08, 1.18), mat_idx=5)
                    _box(bm, (kx, side * (dy * 0.44 + 0.06), 0.4 + h1 + h2 * 0.52), (1.28, 0.06, 1.04), mat_idx=4)
            _gable_roof(bm, wx * 0.88, dy * 0.88, 0.4 + h1 + h2, roof_h, overhang=0.82, mat_roof=1, mat_ridge=2, mat_timber=5)
        else:
            _gable_roof(bm, wx, dy, 0.4 + h1, roof_h, overhang=0.92, mat_roof=1, mat_ridge=2, mat_timber=5)

        m = bpy.data.meshes.new(f"JP_House_{vi}")
        bm.to_mesh(m)
        bm.free()
        for _m in [wall_m, roof_m, mat_ridge, mat_found, mat_win, mat_cedar]:
            m.materials.append(_m)
        lib[f"house_{vi}"] = m

    # 2. Vermilion Shinto Myojin Torii Gate + Stone Lanterns
    bm = bmesh.new()
    for sy in (-1.85, 1.85):
        _box(bm, (0.0, sy, -0.5), (0.68, 0.68, 2.0), mat_idx=2)
        ret = bmesh.ops.create_cone(
            bm, cap_ends=True, segments=16, radius1=0.24, radius2=0.19, depth=5.1,
            matrix=Matrix.Translation((0.0, sy * 0.96, 2.55)) @ Euler((sy * 0.025, 0.0, 0.0)).to_matrix().to_4x4()
        )
        for v in ret["verts"]:
            for f in v.link_faces:
                f.material_index = 0
    _box(bm, (0.0, 0.0, 4.10), (0.24, 5.10, 0.28), mat_idx=0)
    _box(bm, (0.0, 0.0, 4.62), (0.26, 0.44, 0.78), mat_idx=1)
    _box(bm, (0.0, 0.0, 5.05), (0.44, 5.60, 0.26), mat_idx=0)
    _box(bm, (0.0, 0.0, 5.26), (0.54, 5.90, 0.20), mat_idx=1)
    for side in (-1.0, 1.0):
        _box(bm, (0.0, side * 3.15, 5.36), (0.54, 0.72, 0.22), mat_idx=1, rot_eul=(side * 0.24, 0.0, 0.0))
        _box(bm, (0.0, side * 3.05, 5.16), (0.44, 0.65, 0.22), mat_idx=0, rot_eul=(side * 0.24, 0.0, 0.0))
    for sy in (-2.9, 2.9):
        _box(bm, (2.2, sy, 0.55), (0.55, 0.55, 1.5), mat_idx=2)
        _box(bm, (2.2, sy, 1.55), (0.42, 0.42, 0.52), mat_idx=2)
        _box(bm, (2.2, sy, 1.95), (0.76, 0.76, 0.24), mat_idx=2)

    m = bpy.data.meshes.new("JP_Shrine_Torii")
    bm.to_mesh(m)
    bm.free()
    for _m in [mat_vermilion, mat_torii_cap, mat_stone]:
        m.materials.append(_m)
    lib["torii"] = m

    # 3. Traditional Cedar Shinto Shrine Hall (rich cedar wood grain, timber frame, porch, shimenawa & 3D kawara roof ribs!)
    mat_shrine_wood = make_box_pbr_material(
        "JP_Shrine_Cedar_PBR", "cedar_wood", tile_scale=1.15,
        tint_rgb=(0.68, 0.50, 0.36), tint_fac=0.32, normal_strength=1.25,
        rough_mult=0.95, ao_darkening=0.58, ao_dist=0.9, macro_var=0.14
    )
    mat_shrine_roof = make_box_pbr_material(
        "JP_Shrine_Kawara_PBR", "kawara_roof", tile_scale=1.15,
        tint_rgb=(0.30, 0.33, 0.38), tint_fac=0.55, normal_strength=1.35,
        rough_mult=0.68, ao_darkening=0.52, ao_dist=0.7, metallic=0.10, macro_var=0.12
    )
    mat_shrine_dark = _mat("JP_Shrine_DarkTimber", (0.14, 0.085, 0.055), rough=0.72)
    mat_shrine_paper = _mat("JP_Shrine_ShideWhite", (0.88, 0.86, 0.80), rough=0.65)
    mat_shrine_gold = _mat("JP_Shrine_BronzeGold", (0.68, 0.52, 0.22), rough=0.35, metallic=0.85)
    mat_shrine_rope = _mat("JP_Shrine_Shimenawa", (0.58, 0.46, 0.24), rough=0.88)

    bm = bmesh.new()
    # Stepped stone plinth & foundation
    _box(bm, (0.0, 0.0, -0.55), (6.0, 5.0, 2.2), mat_idx=2)
    _box(bm, (-3.15, 0.0, 0.22), (0.95, 2.2, 0.65), mat_idx=2)
    # Main cedar plank hall body
    _box(bm, (0.2, 0.0, 1.80), (4.4, 3.8, 2.5), mat_idx=0)
    # Vertical cedar board-and-batten timber ribs & heavy corner pillars
    for sx in (-2.0, -0.6, 0.8, 2.4):
        for sy in (-1.92, 1.92):
            _box(bm, (sx, sy, 1.80), (0.18, 0.16, 2.52), mat_idx=3)
    for sy in (-1.92, -0.95, 0.95, 1.92):
        _box(bm, (-2.02, sy, 1.80), (0.18, 0.18, 2.54), mat_idx=3)
    # Horizontal lintel beams (nuki) & top eave plate
    for hz in (0.68, 2.68, 3.02):
        _box(bm, (0.2, 0.0, hz), (4.65, 4.08, 0.16), mat_idx=3)
    # Wooden front porch (engawa) facing -X toward the Torii + steps + railing
    _box(bm, (-2.45, 0.0, 0.62), (1.10, 4.10, 0.16), mat_idx=0)
    for step_i in range(3):
        _box(bm, (-3.05 - step_i * 0.24, 0.0, 0.44 - step_i * 0.16), (0.26, 1.85, 0.14), mat_idx=0)
    for sy in (-1.88, 1.88):
        _box(bm, (-2.85, sy, 1.10), (0.14, 0.14, 0.95), mat_idx=3)
        _box(bm, (-2.42, sy, 1.48), (0.98, 0.12, 0.10), mat_idx=3)
    # Recessed slatted wooden shrine doors (koushi-do) + warm interior paper glow + saisen-bako offering box
    _box(bm, (-2.04, 0.0, 1.65), (0.10, 1.72, 1.88), mat_idx=3)
    _box(bm, (-2.07, 0.0, 1.68), (0.06, 1.48, 1.68), mat_idx=7)
    for ky in (-0.52, -0.18, 0.18, 0.52):
        _box(bm, (-2.10, ky, 1.68), (0.05, 0.06, 1.72), mat_idx=3)
    for kz in (1.15, 1.55, 1.95, 2.30):
        _box(bm, (-2.10, 0.0, kz), (0.05, 1.54, 0.06), mat_idx=3)
    # Saisen-bako (slatted wooden offering box) on porch
    _box(bm, (-2.52, 0.0, 1.02), (0.52, 1.15, 0.68), mat_idx=3)
    # Sacred braided straw shimenawa rope + 4 hanging white paper shide zigzags under front eave!
    _box(bm, (-2.38, 0.0, 2.74), (0.22, 2.25, 0.22), mat_idx=6)
    for sy in (-0.75, -0.25, 0.25, 0.75):
        _box(bm, (-2.44, sy, 2.48), (0.04, 0.14, 0.34), mat_idx=4, rot_eul=(0.0, 0.0, 0.15))

    # Gable kawara-tile roof with deep eaves + 3D corrugated tile ribs + exposed wooden rafters (taruki)!
    z_eave, ridge_h = 3.05, 2.05
    _gable_roof(bm, 4.6, 4.0, z_eave, ridge_h, overhang=1.15, mat_roof=1, mat_ridge=5, mat_timber=3)
    hy_r = 4.0 * 0.5 + 1.15
    hx_r = 4.6 * 0.5 + 1.15
    slope_l = math.hypot(hy_r, ridge_h)
    ang_r = math.atan2(ridge_h, hy_r)
    for side in (-1.0, 1.0):
        cy = side * hy_r * 0.5
        cz = z_eave + ridge_h * 0.5
        # 17 crisp 3D kawara tile roll-ribs down each slope!
        for ti in range(17):
            tx = -hx_r + 0.18 + ti * (2.0 * hx_r - 0.36) / 16.0
            _box(bm, (tx, cy, cz + 0.14), (0.09, slope_l + 0.16, 0.09), mat_idx=1, rot_eul=(-side * ang_r, 0.0, 0.0))
        # Exposed wooden rafters (taruki) projecting cleanly under the eaves
        for ri in range(11):
            rx = -hx_r + 0.25 + ri * (2.0 * hx_r - 0.50) / 10.0
            _box(bm, (rx, side * (hy_r - 0.35), z_eave - 0.02), (0.09, 0.85, 0.11), mat_idx=3, rot_eul=(-side * ang_r, 0.0, 0.0))
    # Decorative bronze-gold ridge katsuogi billets & end finials
    for kx in (-1.6, -0.8, 0.0, 0.8, 1.6):
        _box(bm, (kx, 0.0, z_eave + ridge_h + 0.36), (0.20, 0.64, 0.18), mat_idx=5)

    m = bpy.data.meshes.new("JP_Shrine_Hall")
    bm.to_mesh(m)
    bm.free()
    for _m in [mat_shrine_wood, mat_shrine_roof, mat_stone, mat_shrine_dark, mat_shrine_paper, mat_shrine_gold, mat_shrine_rope, mat_win]:
        m.materials.append(_m)
    lib["shrine_hall"] = m

    # 4. Japanese Rural Concrete Utility Pole (denchu — signature visual cue of rural Japan!)
    bm = bmesh.new()
    ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.17, radius2=0.10, depth=10.5,
                                matrix=Matrix.Translation((0.0, 0.0, 4.75)))
    for v in ret["verts"]:
        for f in v.link_faces:
            f.material_index = 0
    for hz in (8.6, 9.4):
        _box(bm, (0.0, 0.0, hz), (0.10, 1.85, 0.10), mat_idx=1)
    # Pole transformer drum
    ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.22, radius2=0.22, depth=0.85,
                                matrix=Matrix.Translation((0.28, 0.0, 7.6)))
    for v in ret["verts"]:
        for f in v.link_faces:
            f.material_index = 1
    m = bpy.data.meshes.new("JP_Utility_Pole")
    bm.to_mesh(m)
    bm.free()
    m.materials.append(mat_found)
    m.materials.append(mat_steel)
    lib["utility_pole"] = m

    # 5. Japanese Rural Compact Car
    bm = bmesh.new()
    _box(bm, (0.0, 0.0, 0.65), (4.3, 1.72, 0.75), mat_idx=0)
    _box(bm, (-0.2, 0.0, 1.25), (2.25, 1.48, 0.55), mat_idx=1)
    for wx in (-1.35, 1.35):
        for wy in (-0.82, 0.82):
            ret = bmesh.ops.create_cone(
                bm, cap_ends=True, segments=12, radius1=0.32, radius2=0.32, depth=0.22,
                matrix=Matrix.Translation((wx, wy, 0.32)) @ Euler((math.pi * 0.5, 0.0, 0.0)).to_matrix().to_4x4()
            )
            for v in ret["verts"]:
                for f in v.link_faces:
                    f.material_index = 1
    m = bpy.data.meshes.new("JP_Road_Car")
    bm.to_mesh(m)
    bm.free()
    m.materials.append(mat_car_w)
    m.materials.append(mat_car_g)
    lib["car"] = m

    return lib
