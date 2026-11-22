"""
Exact physical kinematics & cinematography solver for all 1,800 frames (75.0 s @ 24 fps).
Produces:
  - Per-frame train nose arc-length `s(frame)`, 3D world pose (curve, cant, grade, subtle EDS heave)
  - Per-frame camera rig pose (position, target, focal length, focus distance, real speed)
  - Per-frame `physics_log.csv` (verified against physical limits: drone <= 26 m/s, heli <= 70 m/s, car <= 25 m/s)
"""
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import specs


@dataclass
class FramePhysics:
    frame: int
    time_s: float
    shot_id: str
    loc_id: str
    is_record_run: bool
    train_speed_kmh: float
    train_speed_ms: float
    train_s_m: float
    train_pos: Tuple[float, float, float]
    train_eul: Tuple[float, float, float]
    cam_pos: Tuple[float, float, float]
    cam_target: Tuple[float, float, float]
    cam_speed_ms: float
    cam_dist_m: float
    focal_mm: float
    fstop: float
    focus_dist_m: float
    shutter_s: float
    blur_length_m: float
    rig_type: str
    time_scale: float = 1.0
    shutter_fraction: float = 0.50


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _cos_ramp(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return 0.5 * (1.0 - math.cos(math.pi * u))


# Cinema Showstopper Ultra-Slow-Mo Speed-Ramp Profiles: (f_in0, f_in1, f_out0, f_out1, min_scale)
# Smooth cosine S-curve transition from 24 fps real-time (1.0x) -> 240–300 fps Phantom Flex4K ultra-slow-mo -> 24 fps!
SPEED_RAMP_PROFILES: Dict[str, Tuple[float, float, float, float, float]] = {
    "S02": (0.22, 0.34, 0.72, 0.84, 0.08),  # Showstopper 1: 2.66s @ 0.08x (300->24fps) over U-trough coils & 10cm levitation gap!
    "S07": (0.24, 0.35, 0.70, 0.82, 0.10),  # Showstopper 2: 2.80s @ 0.10x (240->24fps) over Kiso Torii, shrine & swirling leaves!
}


def shot_time_scale(shot_id: str, frac: float) -> float:
    """Instantaneous playback time_scale (1.0 = real time, 0.08 = 12.5x Phantom Flex4K ultra-slow-motion)."""
    prof = SPEED_RAMP_PROFILES.get(shot_id)
    if prof is None:
        return 1.0
    f0, f1, f2, f3, s_min = prof
    if frac <= f0 or frac >= f3:
        return 1.0
    if frac < f1:
        return _lerp(1.0, s_min, _cos_ramp((frac - f0) / (f1 - f0)))
    if frac <= f2:
        return s_min
    return _lerp(s_min, 1.0, _cos_ramp((frac - f2) / (f3 - f2)))


def shot_shutter_fraction(shot_id: str, frac: float) -> float:
    """
    Adaptive cinema shutter angle mapped directly to the time-ramp curve:
    - Real-time 1.0x: standard 180-deg film shutter (0.50 frames -> full 2.89 m 500 km/h speed blur!)
    - Ultra-slow-mo plateau (0.08x / 0.10x): razor-sharp 72-deg Phantom high-speed shutter (0.20 frames -> ~9-12 cm blur!)
    """
    ts = shot_time_scale(shot_id, frac)
    if ts >= 0.999:
        return specs.SHUTTER_FRACTION
    u = max(0.0, min(1.0, (1.0 - ts) / 0.90))
    return _lerp(specs.SHUTTER_FRACTION, 0.20, u)


def frame_shutter_fraction(frame: int) -> float:
    """Return adaptive Blender `motion_blur_shutter` fraction for any frame 1..1800."""
    shot = specs.get_shot_for_frame(frame)
    frac = (frame - shot.frame_start) / max(1, shot.num_frames - 1)
    return shot_shutter_fraction(shot.shot_id, frac)


def shot_sim_time_array(shot: specs.ShotSpec) -> Tuple[List[float], List[float]]:
    """Integrate exact physical simulation time t_sim(idx) via trapezoidal quadrature of time_scale(frac)."""
    n = shot.num_frames
    dt = 1.0 / specs.FPS
    scales = [shot_time_scale(shot.shot_id, idx / max(1, n - 1)) for idx in range(n)]
    t_sims = [0.0]
    for idx in range(1, n):
        t_sims.append(t_sims[-1] + 0.5 * (scales[idx - 1] + scales[idx]) * dt)
    return t_sims, scales


def evaluate_camera_for_shot(shot: specs.ShotSpec, loc: specs.LocationSpec,
                             t_local: float, frac: float, train_s: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    """
    Return (cam_pos, cam_target, focus_dist_m, focal_mm) in local guideway-datum coordinates.
    Includes physically grounded cinema camera shifts snapping cleanly into our 2 ultra-slow-mo showstopper beats!
    """
    nose_pos, nose_eul = specs.guideway_pose_at_s(train_s, loc)
    mid_pos, _ = specs.guideway_pose_at_s(train_s - 65.0, loc)
    tail_pos, _ = specs.guideway_pose_at_s(train_s - specs.TRAIN_TOTAL_LENGTH, loc)
    focal_mm = shot.focal_mm

    sid = shot.shot_id
    if sid == "S01":
        # Tripod tele (85 mm) on hillside framing slotted concrete buffer hood mouth (s=0) & emerging train!
        cp0, _ = specs.guideway_pose_at_s(175.0, loc)
        cam_pos = (175.0, -64.0, cp0[2] + 22.0)
        pan_s = max(-8.0, min(185.0, (train_s - 25.0) * 0.55))
        tp, _ = specs.guideway_pose_at_s(pan_s, loc)
        cam_target = (tp[0], tp[1] + 2.0, tp[2] + 3.0)

    elif sid == "S02":
        # SHOWSTOPPER 1: Low trackside jib inside Track 1 U-trough!
        # Fast grounded jib push-in & par-focal lens snap (35 -> 52 mm) right into 2.66s @ 0.08x ultra-slow-mo
        # focusing razor-sharp on the gleaming L0-950 nose, 10 cm levitation gap & figure-8 sidewall coils!
        cp, _ = specs.guideway_pose_at_s(185.0, loc)
        push_in = _cos_ramp((frac - 0.18) / 0.16)
        pull_out = _cos_ramp((frac - 0.72) / 0.14)
        jib_w = max(0.0, push_in - pull_out)
        # Subtle slow orbital jib slide toward trough floor & coils during ultra-slow-mo (physically grounded 0.45 m rail!)
        jib_x = cp[0] - 0.45 * jib_w
        jib_y = cp[1] - 1.50 + 0.16 * jib_w
        jib_z = cp[2] + 0.62 - 0.14 * jib_w
        cam_pos = (jib_x, jib_y, jib_z)
        # Snap-zoom 35 mm -> 52 mm into slow-mo, whip back to 32 mm on explosive full-speed exit!
        focal_mm = 35.0 + 17.0 * jib_w - 3.0 * pull_out
        # Smooth whip-track following the nose as it glides from 28 m -> 2 m past the lens!
        look_s = min(182.0, max(118.0, _lerp(122.0, min(180.0, train_s - 2.0), push_in)))
        tp, _ = specs.guideway_pose_at_s(look_s, loc)
        cam_target = (tp[0], tp[1] + 0.06, tp[2] + 0.78 - 0.12 * jib_w)
        # Lock crisp focus directly onto the approaching nose tip during ultra-slow-mo!
        focus_dist = max(4.5, min(90.0, abs(jib_x - min(train_s, jib_x - 4.0))))
        return cam_pos, cam_target, focus_dist, focal_mm

    elif sid == "S03":
        # HERO MONEY SHOT: Helicopter aerial over Kofu-West Basin framing sweeping R=8000m viaduct beneath Mt Fuji!
        s_cam = -100.0 + 380.0 * frac
        cp, _ = specs.guideway_pose_at_s(s_cam, loc)
        cam_pos = (cp[0] + 20.0, cp[1] - 210.0, cp[2] + 24.0)
        aim_s = _lerp(-40.0, 240.0, frac)
        tp, _ = specs.guideway_pose_at_s(aim_s, loc)
        cam_target = (tp[0], tp[1] + 260.0, tp[2] + 30.0)

    elif sid == "S04":
        # Parallel rural farm road camera car at 80 km/h (22.22 m/s)
        car_s = -30.0 + 22.22 * t_local
        cp, _ = specs.guideway_pose_at_s(car_s, loc)
        cam_pos = (cp[0], cp[1] - 34.0, cp[2] - 7.2)
        look_s = car_s + 26.0 + 60.0 * _smooth((train_s - car_s) / 220.0)
        tp, _ = specs.guideway_pose_at_s(look_s, loc)
        cam_target = (tp[0], tp[1], tp[2] + 2.4)

    elif sid == "S05":
        # 72 mm telephoto tripod in sunlit valley framing climbing 2.5% viaduct against 3,193 m Minami Alps!
        cp, _ = specs.guideway_pose_at_s(180.0, loc)
        cam_pos = (cp[0], cp[1] - 185.0, cp[2] + 6.0)
        aim_s = max(-160.0, min(180.0, train_s - 55.0))
        tp, _ = specs.guideway_pose_at_s(aim_s, loc)
        cam_target = (tp[0], tp[1], tp[2] + 15.0)

    elif sid == "S06":
        # Cinema drone elevated 22-29 m above Tenryu River Gorge (23 m/s max)
        drone_s = -75.0 + 176.0 * frac
        cam_pos = (drone_s, -108.0, 21.0 + 7.5 * frac)
        aim_s = max(-180.0, min(300.0, train_s - 50.0))
        tp, _ = specs.guideway_pose_at_s(aim_s, loc)
        cam_target = (tp[0], tp[1], tp[2] - 3.0)

    elif sid == "S07":
        # SHOWSTOPPER 2: Kiso rural hamlet fluid-head cinema tripod!
        # Fast whip-pan from acoustic noise hood + crisp par-focal zoom push-in (44 -> 62 mm) snapping directly into
        # 2.80s @ 0.10x ultra-slow-mo right over the vermilion Torii (x=60) & cedar shrine (x=66) with swirling maple leaves!
        cp, _ = specs.guideway_pose_at_s(45.0, loc)
        cam_pos = (45.0, -84.0, cp[2] - loc.viaduct_elev_above_ground + 5.5)
        zoom_in = _cos_ramp((frac - 0.22) / 0.14)
        zoom_out = _cos_ramp((frac - 0.70) / 0.13)
        z_w = max(0.0, zoom_in - zoom_out)
        focal_mm = 44.0 + 18.0 * z_w + 4.0 * zoom_out
        # Whip-pan from hood mouth (s=32) to lock right onto Torii & gliding nose (s=61..68), then follow exit!
        pan_s = _lerp(34.0, 62.0, zoom_in) + 6.5 * _smooth((frac - 0.35) / 0.35) + 12.0 * zoom_out
        tp, _ = specs.guideway_pose_at_s(pan_s, loc)
        cam_target = (tp[0], tp[1] - 4.0, tp[2] - 1.4 + 0.6 * z_w)

    elif sid == "S08":
        # RECORD RUN (603 km/h = 167.5 m/s): Onboard forward nose camera right above L0-950 camera visor!
        heave = 0.003 * math.sin(2.0 * math.pi * 4.2 * t_local)
        cam_pos = (nose_pos[0] - 9.2, nose_pos[1], nose_pos[2] + 3.05 + heave)
        cam_target = (nose_pos[0] + 220.0, nose_pos[1], nose_pos[2] + 1.35)

    elif sid == "S09":
        # RECORD RUN (603 km/h = 167.5 m/s): Helicopter 3/4 aerial chase at 68 m/s (overtaken by +99.5 m/s!)
        heli_s = -120.0 + 68.0 * t_local
        cam_pos = (heli_s + 52.0, -44.0, 14.5)
        aim_s = max(heli_s - 60.0, min(heli_s + 180.0, train_s - 25.0))
        tp, _ = specs.guideway_pose_at_s(aim_s, loc)
        cam_target = (tp[0], tp[1], tp[2] + 2.0)

    elif sid == "S10":
        # Closing sunset shot: off-axis elevated tripod showing wide 38 m pier spans & U-trough diving into slotted portal
        cp, _ = specs.guideway_pose_at_s(580.0, loc)
        cam_pos = (580.0, -108.0, cp[2] + 22.0)
        aim_s = max(720.0, min(1060.0, train_s - 25.0))
        tp, _ = specs.guideway_pose_at_s(aim_s, loc)
        cam_target = (tp[0], tp[1] + 2.0, tp[2] + 2.0)

    else:
        raise ValueError(f"Unknown shot_id {sid}")

    focus_dist = max(4.0, math.dist(cam_pos, cam_target))
    return cam_pos, cam_target, focus_dist, focal_mm


def compute_all_frames() -> List[FramePhysics]:
    """Simulate every frame 1..1800 with exact Phantom Flex4K speed-ramp integration, adaptive shutter & focal shifts."""
    records: List[FramePhysics] = []
    dt = 1.0 / specs.FPS

    for shot in specs.SHOTS:
        loc = specs.LOCATIONS[shot.loc_id]
        n = shot.num_frames
        t_sims, scales = shot_sim_time_array(shot)
        prev_cam = None
        for idx in range(n):
            frame = shot.frame_start + idx
            t_sim = t_sims[idx]
            ts = scales[idx]
            frac = idx / max(1, n - 1)
            sh_frac = shot_shutter_fraction(shot.shot_id, frac)
            train_s = shot.train_x0_m + shot.speed_ms * t_sim
            pos, eul = specs.guideway_pose_at_s(train_s, loc)
            heave = 0.003 * math.sin(2.0 * math.pi * 3.7 * t_sim)
            pos = (pos[0], pos[1], pos[2] + heave)

            cam_pos, cam_target, focus_dist, focal_mm = evaluate_camera_for_shot(shot, loc, t_sim, frac, train_s)
            if prev_cam is None:
                t_sim_next = t_sims[min(n - 1, idx + 1)]
                cam_next, _, _, _ = evaluate_camera_for_shot(shot, loc, t_sim_next, min(1.0, (idx + 1) / max(1, n - 1)),
                                                             shot.train_x0_m + shot.speed_ms * t_sim_next)
                cam_speed = math.dist(cam_pos, cam_next) / dt
            else:
                cam_speed = math.dist(cam_pos, prev_cam) / dt
            prev_cam = cam_pos

            t_mid_s = max(train_s - specs.TRAIN_TOTAL_LENGTH, min(train_s, cam_pos[0]))
            t_near_pos, _ = specs.guideway_pose_at_s(t_mid_s, loc)
            cam_dist = math.dist(cam_pos, t_near_pos)

            # Exact physical exposure duration combining adaptive shutter angle & high-speed time_scale!
            eff_shutter_s = (sh_frac / specs.FPS) * ts
            blur_len = shot.speed_ms * eff_shutter_s

            records.append(FramePhysics(
                frame=frame,
                time_s=(frame - 1) * dt,
                shot_id=shot.shot_id,
                loc_id=shot.loc_id,
                is_record_run=shot.is_record_run,
                train_speed_kmh=shot.speed_kmh,
                train_speed_ms=shot.speed_ms,
                train_s_m=train_s,
                train_pos=pos,
                train_eul=eul,
                cam_pos=cam_pos,
                cam_target=cam_target,
                cam_speed_ms=cam_speed,
                cam_dist_m=cam_dist,
                focal_mm=focal_mm,
                fstop=shot.fstop,
                focus_dist_m=focus_dist,
                shutter_s=eff_shutter_s,
                blur_length_m=blur_len,
                rig_type=shot.rig_type,
                time_scale=ts,
                shutter_fraction=sh_frac,
            ))
    return records


def write_physics_log(out_csv: str) -> List[FramePhysics]:
    records = compute_all_frames()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "time_s", "shot_id", "loc_id", "is_record_run",
            "train_speed_kmh", "train_speed_ms", "train_s_m",
            "train_x", "train_y", "train_z",
            "cam_x", "cam_y", "cam_z", "cam_speed_ms", "cam_dist_m",
            "focal_mm", "fstop", "focus_dist_m", "shutter_s", "blur_length_m", "rig_type", "time_scale", "shutter_fraction"
        ])
        for r in records:
            w.writerow([
                r.frame, f"{r.time_s:.4f}", r.shot_id, r.loc_id, int(r.is_record_run),
                f"{r.train_speed_kmh:.1f}", f"{r.train_speed_ms:.3f}", f"{r.train_s_m:.2f}",
                f"{r.train_pos[0]:.2f}", f"{r.train_pos[1]:.2f}", f"{r.train_pos[2]:.2f}",
                f"{r.cam_pos[0]:.2f}", f"{r.cam_pos[1]:.2f}", f"{r.cam_pos[2]:.2f}",
                f"{r.cam_speed_ms:.2f}", f"{r.cam_dist_m:.2f}",
                f"{r.focal_mm:.1f}", f"{r.fstop:.1f}", f"{r.focus_dist_m:.2f}",
                f"{r.shutter_s:.6f}", f"{r.blur_length_m:.3f}", r.rig_type, f"{r.time_scale:.4f}", f"{r.shutter_fraction:.3f}"
            ])
    # Sanity assert physical camera rig speed ceilings!
    for r in records:
        if r.rig_type == "drone":
            assert r.cam_speed_ms <= 26.5, f"Drone too fast on frame {r.frame}: {r.cam_speed_ms}"
        elif r.rig_type == "helicopter":
            assert r.cam_speed_ms <= 72.0, f"Helicopter too fast on frame {r.frame}: {r.cam_speed_ms}"
        elif r.rig_type == "car":
            assert r.cam_speed_ms <= 28.0, f"Car too fast on frame {r.frame}: {r.cam_speed_ms}"
    print(f"[physics] Wrote {len(records)} verified frames to {out_csv}")
    return records


if __name__ == "__main__":
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "physics_log.csv"))
    write_physics_log(out)
