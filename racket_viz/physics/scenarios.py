"""Defines every precomputed scenario the app can play back.

A scenario is a `mode: "free" | "controlled"` sequence of one or more segments,
each producing a chunk of (t, quaternion, M_body) frames plus display metadata
(label/caption/dot-color). "free" scenarios reuse `rigid_body.integrate_full` with
no torque_fn (Casimir + energy conserved); "controlled" scenarios supply a
controller from `controllers.py` as the torque_fn (neither is conserved — see
ARCHITECTURE.md).
"""
import numpy as np

import controllers
import phase_portrait
import racket_geometry
import rigid_body


def _integrate_segment(w0, I, T, N, torque_fn=None):
    """One segment, always starting from the identity orientation (each segment
    shows its own motion clearly, matching dzhanibekov_display.py's narrative)."""
    omegas, quats = rigid_body.integrate_full(w0, I, T, N, torque_fn=torque_fn)
    M_body = omegas * np.asarray(I, dtype=float)[None, :]
    t = np.linspace(0.0, T, N)
    return t, quats, M_body


def _geometry_block(geometry):
    """Static racket mesh data every scenario needs for rendering (see
    ARCHITECTURE.md's `geometry` schema key) -- the same for every frame, so
    exported once per scenario rather than duplicated per-frame."""
    return {
        "verts_body": geometry["verts_body"],
        "handle_idx": geometry["handle_idx"],
        "hoop_idx": geometry["hoop_idx"],
        "face_thickness": geometry["face_thickness"],
        "face_normal_body": geometry["face_normal_body"],
    }


def _concat_segments(segments):
    """segments: list of (t, quats, M_body, label, caption, dot_color).
    Returns (t, quats, M_body, segment_meta) with t offset to be continuously
    increasing across segments and segment_meta = [{start,end,label,caption,dot_color}]."""
    all_t, all_q, all_M, meta = [], [], [], []
    cursor = 0
    t_offset = 0.0
    for t, quats, M_body, label, caption, dot_color in segments:
        n = len(t)
        all_t.append(t + t_offset)
        all_q.append(quats)
        all_M.append(M_body)
        meta.append({
            "start": cursor, "end": cursor + n,
            "label": label, "caption": caption, "dot_color": dot_color,
        })
        cursor += n
        t_offset += t[-1] if n else 0.0
    return (
        np.concatenate(all_t),
        np.vstack(all_q),
        np.vstack(all_M),
        meta,
    )


def build_free_narrative_scenario(geometry=None, spin_rate=1.5):
    """The 6-segment torque-free narrative from `dzhanibekov_display.py` Section 6:
    stable fixed point -> small orbit -> near-separatrix flip -> separatrix ->
    small orbit -> stable fixed point (other axis)."""
    geometry = geometry or racket_geometry.build_racket()
    I = geometry["I"]
    imin, imid, imax = geometry["imin"], geometry["imid"], geometry["imax"]

    R_cas = phase_portrait.casimir_radius(I, imid, spin_rate)
    H_axis = phase_portrait.energy_at_axes(I, R_cas)
    H_sep = H_axis[imid]
    eps = 0.001 * R_cas
    alpha_small = 0.30

    segs = []

    w_fixed_min = np.zeros(3)
    w_fixed_min[imin] = R_cas / I[imin]
    segs.append((
        *_integrate_segment(w_fixed_min, I, T=2, N=100),
        "Stable spin", "Spinning cleanly about the long axis: no wobble.", "#44ee66",
    ))

    H_small_min = H_axis[imin] + alpha_small * (H_sep - H_axis[imin])
    ic_small_min = phase_portrait.ic_from_H(H_small_min, eps, I, R_cas, imin, imax, imid)
    w_small_min = np.array(ic_small_min) / I
    segs.append((
        *_integrate_segment(w_small_min, I, T=2, N=100),
        "Gentle wobble", "A small nudge off-axis; the racket precesses gently.", "royalblue",
    ))

    M_ns_imin = 0.05 * R_cas
    M_ns_imid = np.sqrt(max(0.0, R_cas ** 2 - M_ns_imin ** 2 - eps ** 2))
    M_ns_ic = np.zeros(3)
    M_ns_ic[imin] = M_ns_imin
    M_ns_ic[imid] = M_ns_imid
    M_ns_ic[imax] = eps
    w_ns = M_ns_ic / I
    segs.append((
        *_integrate_segment(w_ns, I, T=20, N=200),
        "The flip", "Close to the tipping point, the racket suddenly flips over.", "royalblue",
    ))

    M_sep_ic = np.zeros(3)
    M_sep_ic[imin] = eps
    M_sep_ic[imax] = eps
    M_sep_ic[imid] = np.sqrt(max(0.0, R_cas ** 2 - 2 * eps ** 2))
    w_sep = M_sep_ic / I
    segs.append((
        *_integrate_segment(w_sep, I, T=9.5, N=200),
        "Exactly balanced", "Launched exactly on the tipping curve: one flip, then it settles.", "darkorange",
    ))

    H_small_max = H_axis[imax] + alpha_small * (H_sep - H_axis[imax])
    ic_small_max = phase_portrait.ic_from_H(H_small_max, eps, I, R_cas, imin, imax, imid)
    w_small_max = np.array(ic_small_max) / I
    segs.append((
        *_integrate_segment(w_small_max, I, T=2, N=100),
        "Gentle wobble", "The same gentle precession, now about the other stable axis.", "royalblue",
    ))

    w_fixed_max = np.zeros(3)
    w_fixed_max[imax] = R_cas / I[imax]
    segs.append((
        *_integrate_segment(w_fixed_max, I, T=2, N=100),
        "Stable spin", "Spinning cleanly about the handle axis: no wobble.", "#44ee66",
    ))

    t, quats, M_body, segment_meta = _concat_segments(segs)
    background = phase_portrait.build_static_curves(I, R_cas, imin, imax, imid)

    return {
        "meta": {
            "mode": "free",
            "I": I,
            "R_cas": R_cas,
            "segments": segment_meta,
            # H_axis[k] is the energy of the pure-axis equilibrium at index k
            # (so H_axis == [H_imin, H_imid, H_imax] since imin=0, imid=1,
            # imax=2 always in this codebase). H_axis[imid] is the separatrix
            # energy -- panel 4's reference lines for the free/conserved case.
            "H_axis": H_axis,
        },
        "frames": {"t": t, "quaternion": quats, "M_body": M_body},
        "background": background,
        "geometry": _geometry_block(geometry),
    }


def build_controlled_scenario(target_axis, geometry=None, Kp=1.0, spin_rate=2 * np.pi, T=6.0, N=600):
    """A single segment driven by `controllers.PDBodyFrameController` from an
    unstable-axis spin toward `target_axis` (0=imin, 1=imid, 2=imax)."""
    geometry = geometry or racket_geometry.build_racket()
    I = geometry["I"]
    imin, imid, imax = geometry["imin"], geometry["imid"], geometry["imax"]

    R_cas = phase_portrait.casimir_radius(I, imid, spin_rate)

    omega_star = np.zeros(3)
    omega_star[target_axis] = spin_rate
    ctrl = controllers.PDBodyFrameController(omega_star, Kp=Kp)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    w0 = np.zeros(3)
    w0[imid] = spin_rate
    t, quats, M_body = _integrate_segment(w0, I, T, N, torque_fn=torque_fn)

    axis_name = {0: "imin", 1: "imid", 2: "imax"}.get(target_axis, str(target_axis))
    segment_meta = [{
        "start": 0, "end": N,
        "label": "Control on",
        "caption": f"An active controller steers the spin toward axis {axis_name}.",
        "dot_color": "#ffaa00",
    }]
    background = phase_portrait.build_static_curves(I, R_cas, imin, imax, imid)

    return {
        "meta": {
            "mode": "controlled",
            "I": I,
            "R_cas": R_cas,
            "segments": segment_meta,
            "target_axis": target_axis,
            "desired_H": 0.5 * float(np.sum(omega_star ** 2 * I)),
            "desired_L": float(np.linalg.norm(I * omega_star)),
        },
        "frames": {"t": t, "quaternion": quats, "M_body": M_body},
        "background": background,
        "geometry": _geometry_block(geometry),
    }


def build_alignment_scenario(R_star, geometry=None, K_R=1.0, K_p=1.0, spin_rate=2 * np.pi, T=8.0, N=800):
    """A single segment driven by `controllers.GeometricAttitudeController` toward
    a full target orientation R_star (Stage F "alignment")."""
    geometry = geometry or racket_geometry.build_racket()
    I = geometry["I"]
    imin, imid, imax = geometry["imin"], geometry["imid"], geometry["imax"]

    R_cas = phase_portrait.casimir_radius(I, imid, spin_rate)

    R_star = np.asarray(R_star, dtype=float)
    omega_star = np.zeros(3)
    ctrl = controllers.GeometricAttitudeController(R_star, omega_star, K_R=K_R, K_p=K_p)

    def torque_fn(t, w, q):
        return ctrl(rigid_body.quat_to_R(q), w)

    w0 = np.zeros(3)
    w0[imid] = spin_rate
    t, quats, M_body = _integrate_segment(w0, I, T, N, torque_fn=torque_fn)

    segment_meta = [{
        "start": 0, "end": N,
        "label": "Alignment",
        "caption": "The controller brings both orientation and spin to rest at the target.",
        "dot_color": "#ffaa00",
    }]
    background = phase_portrait.build_static_curves(I, R_cas, imin, imax, imid)

    return {
        "meta": {
            "mode": "controlled",
            "I": I,
            "R_cas": R_cas,
            "segments": segment_meta,
            "target_R": R_star,
            "desired_H": 0.5 * float(np.sum(omega_star ** 2 * I)),
            "desired_L": float(np.linalg.norm(I * omega_star)),
        },
        "frames": {"t": t, "quaternion": quats, "M_body": M_body},
        "background": background,
        "geometry": _geometry_block(geometry),
    }
