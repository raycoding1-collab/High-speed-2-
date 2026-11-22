"""
Download verified 100% CC0 photogrammetry PBR textures (2K diffuse + normal + roughness)
and 3D models (conifer trees, granite river boulders) directly from Poly Haven API.
Caches locally in `assets_cache/polyhaven/` (gitignored; everything packs into `film.blend`)
and writes provenance manifest to `docs/cc0_assets_provenance.json`.
"""
import json
import os
import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PH_DIR = os.path.join(REPO_ROOT, "assets_cache", "polyhaven")
PROVENANCE_PATH = os.path.join(REPO_ROOT, "docs", "cc0_assets_provenance.json")

# Photogrammetry CC0 2K PBR texture sets (albedo + normal_gl + roughness)
PBR_TEXTURES = {
    "concrete": "brushed_concrete_03",
    "cliff": "aerial_rocks_02",
    "forest_floor": "forest_leaves_03",
    "valley_grass": "aerial_grass_rock",
    "asphalt": "asphalt_02",
    "kawara_roof": "grey_roof_tiles_02",
    "cedar_wood": "dark_wooden_planks",
    "bark": "chinese_cedar_bark",
}

# Photogrammetry CC0 3D models (glTF/GLB at 1k/2k — ultra clean import into bpy!)
CC0_MODELS = {
    "fir_tree": "fir_tree_01",
    "pine_tree": "pine_tree_01",
    "broadleaf_tree": "tree_small_02",
    "grass_clump": "grass_medium_01",
    "boulder": "boulder_01",
}


def _dl(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 2000:
        return dest
    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "MaglevFilmPipeline/1.0"})
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for ch in r.iter_content(chunk_size=1 << 19):
            if ch:
                f.write(ch)
    os.replace(tmp, dest)
    return dest


def fetch_pbr_texture(asset_id: str, res: str = "2k") -> dict:
    """Download diffuse, nor_gl, and rough maps for a Poly Haven CC0 texture."""
    out_dir = os.path.join(PH_DIR, "textures", asset_id)
    meta_file = os.path.join(out_dir, "maps.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            maps = json.load(f)
        if all(os.path.exists(p) for p in maps.values()):
            return maps

    r = requests.get(f"https://api.polyhaven.com/files/{asset_id}", timeout=30,
                     headers={"User-Agent": "MaglevFilmPipeline/1.0"})
    r.raise_for_status()
    data = r.json()

    maps = {}
    for logical, candidates in [
        ("albedo", ["Diffuse", "col", "albedo"]),
        ("normal", ["nor_gl", "Normal"]),
        ("rough", ["Rough", "roughness", "arm"]),
    ]:
        for key in candidates:
            if key in data and res in data[key]:
                fmts = data[key][res]
                fmt = "jpg" if "jpg" in fmts else ("png" if "png" in fmts else next(iter(fmts)))
                url = fmts[fmt]["url"]
                dest = os.path.join(out_dir, f"{asset_id}_{logical}.{fmt}")
                _dl(url, dest)
                maps[logical] = dest
                break

    os.makedirs(out_dir, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(maps, f, indent=2)
    print(f"[polyhaven] Texture {asset_id}: {list(maps.keys())}")
    return maps


def fetch_cc0_model_gltf(asset_id: str, res: str = "1k") -> str:
    """Download a Poly Haven CC0 3D model as glTF (or Blender .blend if available) with its textures."""
    out_dir = os.path.join(PH_DIR, "models", asset_id)
    blend_path = os.path.join(out_dir, f"{asset_id}.blend")
    if os.path.exists(blend_path) and os.path.getsize(blend_path) > 10_000:
        return blend_path

    r = requests.get(f"https://api.polyhaven.com/files/{asset_id}", timeout=30,
                     headers={"User-Agent": "MaglevFilmPipeline/1.0"})
    r.raise_for_status()
    data = r.json()

    # Prefer native Blender .blend file + included textures (zero shader conversion loss!)
    if "blend" in data:
        avail = data["blend"]
        use_res = res if res in avail else ("2k" if "2k" in avail else next(iter(avail)))
        binfo = avail[use_res]["blend"]
        _dl(binfo["url"], blend_path)
        for rel_path, finfo in binfo.get("include", {}).items():
            inc_dest = os.path.join(out_dir, rel_path)
            _dl(finfo["url"], inc_dest)
        print(f"[polyhaven] Model {asset_id} (.blend @ {use_res}) + {len(binfo.get('include', {}))} textures")
        return blend_path

    raise RuntimeError(f"No .blend available for {asset_id}")


def fetch_all_polyhaven_assets() -> dict:
    manifest = {"license": "CC0 1.0 Universal", "source": "https://polyhaven.com", "textures": {}, "models": {}}
    result = {"textures": {}, "models": {}}

    for role, asset_id in PBR_TEXTURES.items():
        try:
            maps = fetch_pbr_texture(asset_id, res="2k")
            result["textures"][role] = maps
            manifest["textures"][role] = {"asset_id": asset_id, "maps": list(maps.keys())}
        except Exception as e:
            print(f"[polyhaven] WARN texture {role} ({asset_id}) failed: {e}")

    for role, asset_id in CC0_MODELS.items():
        try:
            bp = fetch_cc0_model_gltf(asset_id, res="1k")
            result["models"][role] = bp
            manifest["models"][role] = {"asset_id": asset_id, "file": os.path.basename(bp)}
        except Exception as e:
            print(f"[polyhaven] WARN model {role} ({asset_id}) failed: {e}")

    os.makedirs(os.path.dirname(PROVENANCE_PATH), exist_ok=True)
    with open(PROVENANCE_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    return result


if __name__ == "__main__":
    res = fetch_all_polyhaven_assets()
    print("Downloaded Poly Haven CC0 assets:", {k: list(v.keys()) for k, v in res.items()})
