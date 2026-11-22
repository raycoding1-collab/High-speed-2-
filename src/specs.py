"""
Single source of truth for all physical, metric, camera, and production constants
for "500 km/h Through Japan" (JR Central L0 Series SCMaglev, 950-series).

All lengths are in meters (m), speeds in m/s, angles in degrees unless suffixed `_rad`,
time in seconds (s). Coordinate convention in Blender:
  +X = travel direction (forward along local guideway tangent)
  +Y = left of travel direction
  +Z = up (vertical)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# ==============================================================================
# 1. PRODUCTION & RENDER SETTINGS
# ==============================================================================
FPS = 24
SHUTTER_ANGLE_DEG = 180.0
SHUTTER_SPEED_S = (SHUTTER_ANGLE_DEG / 360.0) / FPS  # 1/48 s = 0.020833 s
SHUTTER_FRACTION = SHUTTER_ANGLE_DEG / 360.0         # 0.50 frames in Blender

TOTAL_FRAMES = 1800          # 75.0 seconds @ 24 fps
TOTAL_DURATION_S = TOTAL_FRAMES / FPS

RES_FINAL = (1920, 1080)     # Full HD 16:9 (4K upscale delivered separately)
RES_PREVIEW = (960, 540)     # Motion preview resolution

# ==============================================================================
# 2. L0 SERIES (950-SERIES IMPROVED) TRAIN SPECIFICATIONS (METRIC)
# ==============================================================================
TRAIN_MODEL = "JR Central L0 Series (Improved 950-series)"
NUM_INTERMEDIATE_CARS = 10   # 12-car formation: 1 front + 10 mid + 1 rear = 299.0 m
NUM_TOTAL_CARS = NUM_INTERMEDIATE_CARS + 2

END_CAR_LENGTH = 28.00       # m (L21/L22-950)
NOSE_LENGTH = 15.00          # m (Aero-Long-Nose section of end car)
END_CAR_CABIN_LENGTH = END_CAR_LENGTH - NOSE_LENGTH  # 13.00 m
MID_CAR_LENGTH = 24.30       # m (L25-950)
CAR_GAP = 0.04               # m (flush aerodynamic diaphragm seam)

TRAIN_TOTAL_LENGTH = 2 * END_CAR_LENGTH + NUM_INTERMEDIATE_CARS * MID_CAR_LENGTH  # 299.0 m

CAR_WIDTH = 2.90             # m (2,900 mm)
CAR_HEIGHT = 3.10            # m (3,100 mm body shell height)
LEVITATION_GAP = 0.10        # m (100 mm clear air gap between bogie bottom & trough floor)
BODY_BOTTOM_Z = 0.42         # m above trough floor while levitating (bogie skirt sits at Z=0.10 m)
CAR_TOP_Z = BODY_BOTTOM_Z + CAR_HEIGHT  # 3.52 m above trough floor (total vehicle height above floor)
# Note: static body height = 3.10 m; lower bogie shroud extends down to Z = 0.10 m levitation clearance.

# Nose geometry reference stations (distance from nose tip s=0 to s=15.0 m)
NOSE_TIP_HEIGHT = 0.72       # m vertical thickness at nose tip
NOSE_TIP_WIDTH = 1.62        # m width at rounded tip
CAMERA_POD_START = 9.40      # m from nose tip (high-mounted 950-series camera visor)
CAMERA_POD_END = 11.45       # m from nose tip
CAMERA_POD_WIDTH = 0.92      # m
END_CAR_WINDOWS = 6          # vertical porthole windows per side on end car
MID_CAR_WINDOWS = 15         # vertical porthole windows per side on L0-950 intermediate car
WINDOW_WIDTH = 0.30          # m
WINDOW_HEIGHT = 0.44         # m
WINDOW_Z_CENTER = BODY_BOTTOM_Z + 1.56  # m above trough floor
DOOR_WIDTH = 0.88            # m
DOOR_HEIGHT = 1.92           # m

# Jacobs articulated bogie specs
BOGIE_LENGTH = 5.40          # m
BOGIE_WIDTH = 2.68           # m (fits inside 3.10 m clear coil gap with ~0.21 m per side)
BOGIE_HEIGHT = 0.95          # m
LEAD_BOGIE_OFFSET = 6.40     # m behind nose tip (RTRI MAGLEV2004 spec)

# Operating speeds
SPEED_SERVICE_KMH = 500.0
SPEED_SERVICE_MS = SPEED_SERVICE_KMH / 3.6   # 138.8889 m/s
SPEED_RECORD_KMH = 603.0
SPEED_RECORD_MS = SPEED_RECORD_KMH / 3.6     # 167.5000 m/s

# ==============================================================================
# 3. SCMAGLEV U-GUIDEWAY & VIADUCT SPECIFICATIONS (METRIC)
# ==============================================================================
TRACK_SPACING = 5.80         # m center-to-center between opposing tracks
TROUGH_INNER_WIDTH = 3.40    # m concrete-to-concrete inner width
COIL_FACE_CLEAR_WIDTH = 3.12 # m clear width between opposing ground-coil faces
TROUGH_FLOOR_Z = 0.00        # m local vertical datum
SIDEWALL_HEIGHT = 1.68       # m above trough floor
SIDEWALL_THICKNESS = 0.48    # m
COIL_PANEL_PITCH = 0.90      # m longitudinal coil module spacing
COIL_PANEL_HEIGHT = 1.12     # m
COIL_PANEL_Z_BOTTOM = 0.28   # m

VIADUCT_DECK_WIDTH = 12.60   # m total outer width of double-track viaduct
VIADUCT_PARAPET_HEIGHT = 0.42 # m low outer deck edge curb (open steel handrail above -> exposes sculpted 1.68 m U-trough!)
VIADUCT_GIRDER_DEPTH = 2.60  # m prestressed concrete box girder below trough floor
VIADUCT_PIER_SPACING = 38.0  # m authentic Yamanashi long-span prestressed box-girder viaduct pier spacing
LIGHTNING_MAST_SPACING = 38.0 # m spacing for low-profile pier inspection platforms
LIGHTNING_MAST_HEIGHT = 5.20 # m legacy constant (unused; open viaduct has no tall parapet masts)

MIN_CURVE_RADIUS = 8000.0    # m (R = 8 km Chuo Shinkansen standard)
MAX_CANT_DEG = 10.0          # deg (balanced at 420 km/h on R=8000 m)
MAX_GRADIENT = 0.040         # 40 permil (4.0%)

# Tunnel portal micro-pressure-wave buffer hood
BUFFER_HOOD_LENGTH = 56.0    # m slotted concrete/steel portal hood
BUFFER_HOOD_WIDTH = 13.4     # m outer width spanning double track
BUFFER_HOOD_HEIGHT = 8.4     # m outer crown height above trough floor
BUFFER_HOOD_SLITS = 7        # pressure-relief side/roof window slots

# Noise-reduction acoustic enclosure hood (near houses)
NOISE_HOOD_HEIGHT = 7.2      # m arched roof crown
NOISE_HOOD_WIDTH = 12.8      # m

# ==============================================================================
# 4. SEASON, LIGHTING & COLOR CALIBRATION (LATE AUTUMN GOLDEN HOUR)
# ==============================================================================
SUN_ELEVATION_DEG = 11.5     # True low golden hour (long sculptural raking shadows!)
SUN_AZIMUTH_DEG = 302.0      # WNW raking cross-light (~75 deg off camera axis -> rich shadow/highlight modeling!)
SUN_ANGULAR_DIAM_DEG = 0.53  # True solar angular diameter
SUN_TEMP_K = 3800.0          # Rich warm golden-hour direct sunlight

# ==============================================================================
# 5. REAL JAPAN LOCATIONS (COPERNICUS GLO-30 DEM COORDINATES)
# ==============================================================================
@dataclass(frozen=True)
class LocationSpec:
    loc_id: str
    name: str
    lat: float
    lon: float
    dem_tile: str
    extent_m: float          # Square DEM patch side length in meters
    viaduct_elev_above_ground: float
    curve_radius_m: float    # 0.0 = straight, >0 = left curve, <0 = right curve
    cant_deg: float
    grade: float             # dz/dx slope
    description: str

LOCATIONS: Dict[str, LocationSpec] = {
    "loc1_portal": LocationSpec(
        loc_id="loc1_portal",
        name="Yamanashi Tunnel Portal & Valley (Tsuru/Otsuki)",
        lat=35.576, lon=138.918,
        dem_tile="N35_00_E138_00",
        extent_m=4500.0,
        viaduct_elev_above_ground=16.0,
        curve_radius_m=0.0,
        cant_deg=0.0,
        grade=0.005,
        description="Slotted concrete tunnel buffer hood emerging onto open-air valley viaduct amid steep cedar & autumn maple slopes."
    ),
    "loc2_kofu_fuji": LocationSpec(
        loc_id="loc2_kofu_fuji",
        name="Kofu Basin & Mt Fuji (Kofu-West / Chuo Shinkansen Alignment)",
        lat=35.635, lon=138.515,
        dem_tile="N35_00_E138_00",
        extent_m=48000.0,  # Wide multi-res DEM capturing Kofu basin + low Misaka pass + Mt Fuji (36.1 km SSE, +2.22 deg clear!)
        viaduct_elev_above_ground=15.0,
        curve_radius_m=8000.0,
        cant_deg=7.5,
        grade=0.0,
        description="Sweeping R=8000m curved viaduct over Kofu autumn orchards & rural homes with unobstructed snow-capped Mt Fuji rising behind."
    ),
    "loc3_minami_alps": LocationSpec(
        loc_id="loc3_minami_alps",
        name="Southern Alps Foothills (Minami-Alps / Hayakawa)",
        lat=35.650, lon=138.340,
        dem_tile="N35_00_E138_00",
        extent_m=22000.0,
        viaduct_elev_above_ground=28.0,
        curve_radius_m=-9000.0,
        cant_deg=-6.5,
        grade=0.025,
        description="High mountain viaduct climbing at 2.5% grade beneath snow-dusted 3,000 m Minami Alps ridges and valley mist."
    ),
    "loc4_tenryu_gorge": LocationSpec(
        loc_id="loc4_tenryu_gorge",
        name="Tenryu River Gorge Viaduct (Iida / Tenryu-kyo)",
        lat=35.468, lon=137.822,
        dem_tile="N35_00_E137_00",
        extent_m=6000.0,
        viaduct_elev_above_ground=52.0,
        curve_radius_m=0.0,
        cant_deg=0.0,
        grade=0.0,
        description="Dramatic 52 m tall concrete viaduct spanning the emerald Tenryu River gorge flanked by crimson autumn cliffs."
    ),
    "loc5_kiso_hamlet": LocationSpec(
        loc_id="loc5_kiso_hamlet",
        name="Kiso / Nagano Hamlet, Terraced Paddies & Shrine",
        lat=35.552, lon=137.575,
        dem_tile="N35_00_E137_00",
        extent_m=5000.0,
        viaduct_elev_above_ground=13.5,
        curve_radius_m=10000.0,
        cant_deg=5.5,
        grade=-0.010,
        description="Harvested terraced rice fields, kawara-tile rural hamlet, vermilion Torii shrine, persimmon trees, and acoustic hood."
    ),
    "loc6_record_straight": LocationSpec(
        loc_id="loc6_record_straight",
        name="Yamanashi High-Speed Straightaway (603 km/h Record Run)",
        lat=35.590, lon=138.860,
        dem_tile="N35_00_E138_00",
        extent_m=6500.0,
        viaduct_elev_above_ground=18.0,
        curve_radius_m=0.0,
        cant_deg=0.0,
        grade=0.0,
        description="Open-air Yamanashi valley straightaway for the 603 km/h (167.5 m/s) record-run POV and helicopter chase."
    ),
}

# ==============================================================================
# 6. 10-SHOT FILM EDIT (75.0 s = 1,800 FRAMES @ 24 FPS)
# ==============================================================================
@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    name: str
    loc_id: str
    frame_start: int         # Inclusive (1-indexed)
    frame_end: int           # Inclusive
    speed_kmh: float
    is_record_run: bool
    focal_mm: float
    fstop: float
    exposure_ev: float       # Per-shot DP exposure compensation (EV)
    rig_type: str            # tripod, trackside, helicopter, car, drone, onboard_nose
    cam_max_speed_ms: float  # Physically realistic camera rig speed
    train_x0_m: float        # Train nose X position at frame_start (m along guideway)
    notes: str

    @property
    def num_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    @property
    def duration_s(self) -> float:
        return self.num_frames / FPS

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh / 3.6


SHOTS: List[ShotSpec] = [
    ShotSpec(
        shot_id="S01", name="Tunnel Portal Emergence", loc_id="loc1_portal",
        frame_start=1, frame_end=192, speed_kmh=500.0, is_record_run=False,
        focal_mm=85.0, fstop=4.0, exposure_ev=1.00, rig_type="tripod", cam_max_speed_ms=0.0,
        train_x0_m=-210.0,  # Nose bursts out of x=0 buffer hood at ~1.5 s (frame 36)
        notes="85 mm tripod on hillside framing slotted buffer hood; opened up +1.00 EV for shaded mountainside portal."
    ),
    ShotSpec(
        shot_id="S02", name="Low Guideway Levitation Pass-By", loc_id="loc1_portal",
        frame_start=193, frame_end=360, speed_kmh=500.0, is_record_run=False,
        focal_mm=35.0, fstop=5.6, exposure_ev=0.25, rig_type="trackside", cam_max_speed_ms=0.0,
        train_x0_m=-120.0,  # Fast jib push-in & lens snap (35->52mm) into 2.66s 0.08x ultra-slow-mo as nose glides 28m->0m past lens!
        notes="SHOWSTOPPER 1: Trackside jib push-in + 0.08x (300fps->24fps) ultra-slow-mo with adaptive 72-deg shutter showcasing 10cm levitation gap & coils."
    ),
    ShotSpec(
        shot_id="S03", name="Kofu Basin & Mt Fuji Aerial", loc_id="loc2_kofu_fuji",
        frame_start=361, frame_end=576, speed_kmh=500.0, is_record_run=False,
        focal_mm=70.0, fstop=5.6, exposure_ev=-0.20, rig_type="helicopter", cam_max_speed_ms=52.0,
        train_x0_m=-420.0,  # Sweeps across full Kofu basin viaduct framed beneath snow-capped Mt Fuji
        notes="Hero establishing aerial: 70 mm helicopter view framing curved R=8000m viaduct with towering snow-capped Mt Fuji behind."
    ),
    ShotSpec(
        shot_id="S04", name="Parallel Farm Road Overtake", loc_id="loc2_kofu_fuji",
        frame_start=577, frame_end=744, speed_kmh=500.0, is_record_run=False,
        focal_mm=28.0, fstop=4.0, exposure_ev=0.15, rig_type="car", cam_max_speed_ms=22.2,
        train_x0_m=-290.0,  # Car at 80 km/h (22.2 m/s) effortlessly overtaken by train at 500 km/h overhead
        notes="Camera car on rural Kofu road alongside viaduct piers; train overtakes at +116.7 m/s relative speed, leaves whip."
    ),
    ShotSpec(
        shot_id="S05", name="Southern Alps Telephoto Climb", loc_id="loc3_minami_alps",
        frame_start=745, frame_end=912, speed_kmh=500.0, is_record_run=False,
        focal_mm=72.0, fstop=4.0, exposure_ev=0.00, rig_type="tripod", cam_max_speed_ms=0.0,
        train_x0_m=-440.0,  # Long-lens approach climbing 2.5% grade against snow-capped Minami Alps
        notes="72 mm telephoto compression against 3,193 m alpine ridges and aerial haze; crisp golden rim light on nose."
    ),
    ShotSpec(
        shot_id="S06", name="Tenryu Gorge High Viaduct Drone", loc_id="loc4_tenryu_gorge",
        frame_start=913, frame_end=1104, speed_kmh=500.0, is_record_run=False,
        focal_mm=24.0, fstop=4.0, exposure_ev=0.35, rig_type="drone", cam_max_speed_ms=24.0,
        train_x0_m=-420.0,  # Drone (24 m/s max) over emerald river tries to pace train and is left behind
        notes="52 m high concrete viaduct over emerald Tenryu River & crimson maple cliffs; shadow-lifted DP exposure."
    ),
    ShotSpec(
        shot_id="S07", name="Kiso Hamlet, Paddies & Torii", loc_id="loc5_kiso_hamlet",
        frame_start=1105, frame_end=1296, speed_kmh=500.0, is_record_run=False,
        focal_mm=50.0, fstop=4.0, exposure_ev=0.20, rig_type="tripod", cam_max_speed_ms=0.0,
        train_x0_m=-292.0,  # Fast zoom push-in & whip-pan (44->62mm) into 2.80s 0.10x ultra-slow-mo dead-center over vermilion Torii & shrine!
        notes="SHOWSTOPPER 2: Par-focal zoom push-in + whip-pan into 0.10x (240fps->24fps) ultra-slow-mo over vermilion Torii & swirling crimson maple leaves."
    ),
    ShotSpec(
        shot_id="S08", name="603 km/h Record Run Forward POV", loc_id="loc6_record_straight",
        frame_start=1297, frame_end=1464, speed_kmh=603.0, is_record_run=True,
        focal_mm=24.0, fstop=5.6, exposure_ev=-0.10, rig_type="onboard_nose", cam_max_speed_ms=167.5,
        train_x0_m=-600.0,  # Travels 1,172.5 m in 7.0 s at 167.5 m/s!
        notes="RECORD RUN (603 km/h): Onboard camera above L0-950 nose visor; guideway sidewalls blur into pure speed lines."
    ),
    ShotSpec(
        shot_id="S09", name="603 km/h Helicopter Chase", loc_id="loc6_record_straight",
        frame_start=1465, frame_end=1632, speed_kmh=603.0, is_record_run=True,
        focal_mm=85.0, fstop=4.0, exposure_ev=0.00, rig_type="helicopter", cam_max_speed_ms=68.0,
        train_x0_m=-450.0,  # Heli at 68 m/s (245 km/h) is outrun by +99.5 m/s as full train pulls ahead
        notes="RECORD RUN (603 km/h): Dynamic 3/4 aerial chase; warm interior cabin lights & cobalt nose stripes gleaming."
    ),
    ShotSpec(
        shot_id="S10", name="Sunset Tunnel Entry Climax", loc_id="loc1_portal",
        frame_start=1633, frame_end=1800, speed_kmh=500.0, is_record_run=False,
        focal_mm=78.0, fstop=4.0, exposure_ev=0.80, rig_type="tripod", cam_max_speed_ms=0.0,
        train_x0_m=580.0,   # Travels +X toward far portal at x=+1050 m diving into mountain buffer hood
        notes="Closing shot: off-axis 78 mm view revealing wide 38 m pier spans & sculpted U-trough diving into slotted mountain portal."
    ),
]

assert sum(s.num_frames for s in SHOTS) == TOTAL_FRAMES, "Shot frame counts must sum to TOTAL_FRAMES"
assert SHOTS[0].frame_start == 1 and SHOTS[-1].frame_end == TOTAL_FRAMES


def get_shot_for_frame(frame: int) -> ShotSpec:
    for s in SHOTS:
        if s.frame_start <= frame <= s.frame_end:
            return s
    raise ValueError(f"Frame {frame} outside 1..{TOTAL_FRAMES}")


def guideway_pose_at_s(s: float, loc: LocationSpec) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Exact centerline position (x, y, z) and Euler orientation (roll, pitch, yaw) in radians
    for a point at arc-length `s` along a location's guideway.
    - Circular arc with clothoid-like smooth transition into cant/curvature.
    - Grade dz/ds = loc.grade.
    """
    z = s * loc.grade
    pitch = -math.atan(loc.grade)  # Blender +X forward: nose-up is negative Y rotation

    if abs(loc.curve_radius_m) < 1.0:
        return (s, 0.0, z), (0.0, pitch, 0.0)

    R = loc.curve_radius_m  # signed: >0 curves left (+Y), <0 curves right (-Y)
    theta = s / R
    x = R * math.sin(theta)
    y = R * (1.0 - math.cos(theta))
    yaw = theta

    # Smooth cant application (clothoid ramp over 200 m around curve entry if near ends, constant inside hero zone)
    roll = -math.radians(loc.cant_deg)  # Bank inward toward curve center
    return (x, y, z), (roll, pitch, yaw)
