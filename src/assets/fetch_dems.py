"""
Download real Copernicus GLO-30 30 m DEM tiles for Japan (Chuo Shinkansen corridor)
and a verified CC0 golden-hour outdoor HDRI from Poly Haven.
Caches locally in `assets_cache/` (gitignored; textures get packed into film.blend).
"""
import os
import json
import math
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "assets_cache")
DEM_DIR = os.path.join(CACHE_DIR, "dem")
HDRI_DIR = os.path.join(CACHE_DIR, "hdri")

COPERNICUS_TILES = {
    "N35_00_E138_00": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N35_00_E138_00_DEM/Copernicus_DSM_COG_10_N35_00_E138_00_DEM.tif",
    "N35_00_E137_00": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N35_00_E137_00_DEM/Copernicus_DSM_COG_10_N35_00_E137_00_DEM.tif",
}

# Verified CC0 Poly Haven golden-hour clear/partly-cloudy sky HDRIs (true ~12-18 deg sun elevation!)
POLYHAVEN_CANDIDATES = [
    "kloppenheim_06_puresky",
    "citrus_orchard_puresky",
    "lonely_road_afternoon_puresky",
    "kloofendal_48d_partly_cloudy_puresky",
]


def download_file(url: str, dest: str, timeout: int = 120) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return dest
    print(f"[fetch] Downloading {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "MaglevFilmPipeline/1.0"})
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)
    print(f"[fetch] Saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def fetch_all_dems() -> dict:
    paths = {}
    for tile_id, url in COPERNICUS_TILES.items():
        dest = os.path.join(DEM_DIR, f"{tile_id}.tif")
        paths[tile_id] = download_file(url, dest)
    return paths


def fetch_cc0_hdri(resolution: str = "4k") -> str:
    """Fetch a verified CC0 golden-hour HDRI from Poly Haven API."""
    os.makedirs(HDRI_DIR, exist_ok=True)
    for asset_id in POLYHAVEN_CANDIDATES:
        dest = os.path.join(HDRI_DIR, f"{asset_id}_{resolution}.exr")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            return dest
        try:
            meta_url = f"https://api.polyhaven.com/files/{asset_id}"
            r = requests.get(meta_url, timeout=30, headers={"User-Agent": "MaglevFilmPipeline/1.0"})
            if r.status_code != 200:
                continue
            data = r.json()
            hdri_info = data.get("hdri", {}).get(resolution, {})
            fmt = "exr" if "exr" in hdri_info else ("hdr" if "hdr" in hdri_info else None)
            if not fmt:
                continue
            dl_url = hdri_info[fmt]["url"]
            dest = os.path.join(HDRI_DIR, f"{asset_id}_{resolution}.{fmt}")
            download_file(dl_url, dest)
            # Record provenance
            with open(os.path.join(HDRI_DIR, "hdri_provenance.json"), "w") as f:
                json.dump({"asset_id": asset_id, "license": "CC0", "source": "Poly Haven", "url": dl_url, "file": os.path.basename(dest)}, f, indent=2)
            return dest
        except Exception as e:
            print(f"[fetch] Poly Haven candidate {asset_id} failed: {e}")
    raise RuntimeError("Could not download any CC0 HDRI from Poly Haven")


def sample_dem_patch(lat: float, lon: float, extent_x_m: float, extent_y_m: float,
                     nx: int = 256, ny: int = 256, yaw_deg: float = 0.0,
                     center_offset_xy: tuple = (0.0, 0.0)) -> tuple:
    """
    Extract a metric elevation grid (Z in meters, shape [ny, nx]) centered around (lat, lon)
    from the downloaded Copernicus GLO-30 tiles (seamlessly spanning E137/E138 if needed).
    Returns (X_grid, Y_grid, Z_grid, origin_elev_m).
    X runs [-extent_x/2, +extent_x/2] + offset_x; Y runs [-extent_y/2, +extent_y/2] + offset_y.
    """
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))

    xs = np.linspace(-extent_x_m / 2.0, extent_x_m / 2.0, nx) + center_offset_xy[0]
    ys = np.linspace(-extent_y_m / 2.0, extent_y_m / 2.0, ny) + center_offset_xy[1]
    XX, YY = np.meshgrid(xs, ys)

    # Rotate local (X, Y) into East/North geographic offsets by yaw_deg (0 = +X is East, +Y is North)
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    east_m = XX * c - YY * s
    north_m = XX * s + YY * c

    lons = lon + east_m / m_per_deg_lon
    lats = lat + north_m / m_per_deg_lat

    Z = np.full((ny, nx), np.nan, dtype=np.float32)

    for tile_id in COPERNICUS_TILES:
        tif_path = os.path.join(DEM_DIR, f"{tile_id}.tif")
        if not os.path.exists(tif_path):
            continue
        with rasterio.open(tif_path) as src:
            b = src.bounds
            mask = (lons >= b.left) & (lons <= b.right) & (lats >= b.bottom) & (lats <= b.top)
            if not np.any(mask):
                continue
            # Read tight bounding window for speed
            lo_min, lo_max = float(lons[mask].min()) - 0.01, float(lons[mask].max()) + 0.01
            la_min, la_max = float(lats[mask].min()) - 0.01, float(lats[mask].max()) + 0.01
            lo_min = max(b.left, lo_min)
            lo_max = min(b.right, lo_max)
            la_min = max(b.bottom, la_min)
            la_max = min(b.top, la_max)
            win = from_bounds(lo_min, la_min, lo_max, la_max, src.transform)
            arr = src.read(1, window=win).astype(np.float32)
            wt = src.window_transform(win)
            # Bilinear sample into arr
            inv = ~wt
            cols, rows = inv * (lons[mask], lats[mask])
            r0 = np.clip(np.floor(rows).astype(int), 0, arr.shape[0] - 2)
            c0 = np.clip(np.floor(cols).astype(int), 0, arr.shape[1] - 2)
            dr = np.clip(rows - r0, 0.0, 1.0)
            dc = np.clip(cols - c0, 0.0, 1.0)
            val = (
                arr[r0, c0] * (1 - dr) * (1 - dc)
                + arr[r0, c0 + 1] * (1 - dr) * dc
                + arr[r0 + 1, c0] * dr * (1 - dc)
                + arr[r0 + 1, c0 + 1] * dr * dc
            )
            Z[mask] = val

    if np.any(np.isnan(Z)):
        valid_med = float(np.nanmedian(Z)) if not np.all(np.isnan(Z)) else 400.0
        Z = np.nan_to_num(Z, nan=valid_med)

    # Elevation at local origin (0, 0)
    idx_y = int(np.argmin(np.abs(ys)))
    idx_x = int(np.argmin(np.abs(xs)))
    origin_elev = float(Z[idx_y, idx_x])
    return XX, YY, Z, origin_elev


if __name__ == "__main__":
    dems = fetch_all_dems()
    hdri = fetch_cc0_hdri("4k")
    print("DEMs:", dems)
    print("HDRI:", hdri)
    # Quick sanity check on Mt Fuji summit elevation (35.3606N, 138.7274E -> ~3776m)
    _, _, Zf, e0 = sample_dem_patch(35.3606, 138.7274, 4000.0, 4000.0, nx=64, ny=64)
    print(f"Mt Fuji DEM check: max={Zf.max():.1f} m, min={Zf.min():.1f} m (real summit: 3776 m)")
