"""Tennis-racket geometry → mass distribution → principal-axis inertia tensor.

Adapted from `summer-2026/dzhanibekov_display.py` (Sections 2 & 3): a handle segment
plus an elliptical hoop, uniformly-massed points on each, inertia tensor computed
about the center of mass and diagonalized to find the principal axes.
"""
import numpy as np


def build_racket(
    handle_length=1.0,
    hoop_length=2.0,
    hoop_width=1.0,
    total_mass=1.0,
    hoop_fraction=0.60,
    face_thickness=0.04,
    num_handle_pts=40,
    num_hoop_pts=120,
):
    """Build racket geometry and its principal-axis inertia tensor.

    Returns a dict with:
      verts_body   : (N, 3) vertex positions in the principal-axis (body) frame
      handle_idx   : indices into verts_body for the handle
      hoop_idx     : indices into verts_body for the hoop
      I            : (I1, I2, I3), I1 < I2 < I3
      evecs        : (3, 3) columns are principal axes in the original geometric frame
      imin,imid,imax : int, indices 0/1/2 identifying the stable/unstable/stable axes
      face_thickness : float, passed through for rendering the two racket faces
      face_normal_body : (3,) unit normal to the racket face, in the body frame --
        exactly the geometric [0,0,1] normal re-expressed in the principal-axis
        basis. Guaranteed to equal the imax axis (up to sign): the racket is
        planar (every vertex has z=0 before the principal-axis rotation), and
        for any planar mass distribution the out-of-plane axis is automatically
        a principal axis with the *largest* moment (perpendicular axis theorem:
        I_z = I_x + I_y >= max(I_x, I_y)).
    """
    handle_fraction = 1.0 - hoop_fraction
    hoop_radius_x = hoop_length / 2
    hoop_radius_y = hoop_width / 2
    hoop_center_x = hoop_radius_x

    handle_x = np.linspace(-handle_length, 0.0, num_handle_pts)
    handle_points = np.column_stack([
        handle_x, np.zeros(num_handle_pts), np.zeros(num_handle_pts),
    ])

    theta = np.linspace(0, 2 * np.pi, num_hoop_pts, endpoint=False)
    hoop_points = np.column_stack([
        hoop_center_x + hoop_radius_x * np.cos(theta),
        hoop_radius_y * np.sin(theta),
        np.zeros(num_hoop_pts),
    ])

    vertices_raw = np.vstack([handle_points, hoop_points])
    handle_idx = np.arange(num_handle_pts)
    hoop_idx = np.arange(num_handle_pts, num_handle_pts + num_hoop_pts)

    masses = np.zeros(len(vertices_raw))
    masses[handle_idx] = (handle_fraction * total_mass) / len(handle_idx)
    masses[hoop_idx] = (hoop_fraction * total_mass) / len(hoop_idx)

    com = np.sum(vertices_raw * masses[:, None], axis=0) / total_mass
    verts_c = vertices_raw - com

    I_tensor = np.zeros((3, 3))
    for r, m in zip(verts_c, masses):
        r2 = np.dot(r, r)
        I_tensor += m * (r2 * np.eye(3) - np.outer(r, r))

    evals, evecs = np.linalg.eigh(I_tensor)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # eigh's sign/handedness is arbitrary; flip the last axis if needed so the
    # principal frame is a proper (det=+1) rotation of the geometric frame.
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1

    verts_body = verts_c @ evecs
    face_normal_body = evecs.T @ np.array([0.0, 0.0, 1.0])
    face_normal_body /= np.linalg.norm(face_normal_body)

    return {
        "verts_body": verts_body,
        "handle_idx": handle_idx,
        "hoop_idx": hoop_idx,
        "I": evals,
        "evecs": evecs,
        "imin": 0,
        "imid": 1,
        "imax": 2,
        "face_thickness": face_thickness,
        "face_normal_body": face_normal_body,
        "com": com,
    }
