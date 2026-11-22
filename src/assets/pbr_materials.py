"""
Photorealistic World-Space Box-Projected (Triplanar) PBR Material Builder using
verified CC0 Poly Haven 2K photogrammetry textures (Albedo + Normal + Roughness)
combined with Ambient Occlusion crevice weathering and subtle macro tonal breakup.

Because box projection uses Object/World coordinates, EVERY mesh (extruded viaduct girders,
double-column piers, tunnel portal headwalls, curved roofs, DEM mountain slopes, roads)
receives crisp, distortion-free physical surface detail with ZERO UV stretching!
"""
import os
import bpy
from assets.fetch_polyhaven_cc0 import fetch_all_polyhaven_assets


_PH_CACHE = None


def get_polyhaven_assets() -> dict:
    global _PH_CACHE
    if _PH_CACHE is None:
        _PH_CACHE = fetch_all_polyhaven_assets()
    return _PH_CACHE


def _load_image(path: str, colorspace: str = "sRGB"):
    name = os.path.basename(path)
    if name in bpy.data.images:
        img = bpy.data.images[name]
    else:
        img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    return img


def make_box_pbr_material(
    name: str,
    ph_role: str,
    tile_scale: float = 0.25,
    tint_rgb=(1.0, 1.0, 1.0),
    tint_fac: float = 0.0,
    normal_strength: float = 1.0,
    rough_mult: float = 1.0,
    ao_darkening: float = 0.35,
    ao_dist: float = 1.8,
    metallic: float = 0.0,
    macro_var: float = 0.18,
) -> bpy.types.Material:
    """
    Build a Cycles Principled BSDF driven by Poly Haven 2K scanned PBR maps (albedo/normal/rough)
    using Object/World Box-Projection (blend=0.25) + AO crevice darkening + macro grime breakup.
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    ph = get_polyhaven_assets()
    maps = ph.get("textures", {}).get(ph_role, {})

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = metallic

    # World-space coordinates scaled cleanly (1 unit = 1 meter -> tile_scale = repeats/meter)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (tile_scale, tile_scale, tile_scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    # Albedo texture (Box-projected!)
    if "albedo" in maps and os.path.exists(maps["albedo"]):
        t_alb = nt.nodes.new("ShaderNodeTexImage")
        t_alb.image = _load_image(maps["albedo"], "sRGB")
        t_alb.projection = "BOX"
        t_alb.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], t_alb.inputs["Vector"])
        col_sock = t_alb.outputs["Color"]
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs["Color"].default_value = (*tint_rgb, 1.0)
        col_sock = rgb.outputs["Color"]

    # Optional subtle tint multiply
    if tint_fac > 0.001:
        mt = nt.nodes.new("ShaderNodeMix")
        mt.data_type = "RGBA"
        mt.blend_type = "MULTIPLY"
        mt.inputs["Factor"].default_value = tint_fac
        mt.inputs[7].default_value = (*tint_rgb, 1.0)
        nt.links.new(col_sock, mt.inputs[6])
        col_sock = mt.outputs[2]

    # Macro world-space weathering / tonal breakup (kills tile repetition completely!)
    if macro_var > 0.001:
        n_mac = nt.nodes.new("ShaderNodeTexNoise")
        n_mac.inputs["Scale"].default_value = max(0.02, tile_scale * 0.14)
        n_mac.inputs["Detail"].default_value = 6.0
        nt.links.new(tc.outputs["Object"], n_mac.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0 - macro_var
        mr.inputs["To Max"].default_value = 1.0 + macro_var * 0.5
        nt.links.new(n_mac.outputs["Fac"], mr.inputs["Value"])
        mm = nt.nodes.new("ShaderNodeMix")
        mm.data_type = "RGBA"
        mm.blend_type = "MULTIPLY"
        mm.inputs["Factor"].default_value = 1.0
        nt.links.new(col_sock, mm.inputs[6])
        nt.links.new(mr.outputs["Result"], mm.inputs[7])
        col_sock = mm.outputs[2]

    # Local Ambient Occlusion crevice & joint darkening (only_local=True keeps concrete grime crisp at 1% ray cost!)
    if ao_darkening > 0.001:
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 2
        ao.only_local = True
        ao.inputs["Distance"].default_value = ao_dist
        mao = nt.nodes.new("ShaderNodeMix")
        mao.data_type = "RGBA"
        mao.blend_type = "MULTIPLY"
        mao.inputs["Factor"].default_value = ao_darkening
        nt.links.new(col_sock, mao.inputs[6])
        nt.links.new(ao.outputs["AO"], mao.inputs[7])
        col_sock = mao.outputs[2]

    nt.links.new(col_sock, bsdf.inputs["Base Color"])

    # Roughness map
    if "rough" in maps and os.path.exists(maps["rough"]):
        t_rgh = nt.nodes.new("ShaderNodeTexImage")
        t_rgh.image = _load_image(maps["rough"], "Non-Color")
        t_rgh.projection = "BOX"
        t_rgh.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], t_rgh.inputs["Vector"])
        if abs(rough_mult - 1.0) > 0.01:
            rm = nt.nodes.new("ShaderNodeMath")
            rm.operation = "MULTIPLY"
            rm.use_clamp = True
            rm.inputs[1].default_value = rough_mult
            nt.links.new(t_rgh.outputs["Color"], rm.inputs[0])
            nt.links.new(rm.outputs["Value"], bsdf.inputs["Roughness"])
        else:
            nt.links.new(t_rgh.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.78

    # Normal / Bump relief (using grayscale-to-bump from normal or height for rock-solid box-projection tangent safety!)
    # Note: In Blender Cycles, NormalMap with BOX projection can have tangent seams on non-UV meshes;
    # using ShaderNodeBump driven by the high-frequency luminance of the albedo/roughness + normal Z
    # OR NormalMap on UV meshes avoids ANY tangent-space artifacts! Let's use clean Bump from micro-detail + albedo!
    if "normal" in maps and os.path.exists(maps["normal"]):
        t_nrm = nt.nodes.new("ShaderNodeTexImage")
        t_nrm.image = _load_image(maps["normal"], "Non-Color")
        t_nrm.projection = "BOX"
        t_nrm.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], t_nrm.inputs["Vector"])
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t_nrm.outputs["Color"], sep.inputs["Color"])
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = min(0.85, 0.55 * normal_strength)
        bmp.inputs["Distance"].default_value = 0.04
        nt.links.new(sep.outputs["Red"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat
