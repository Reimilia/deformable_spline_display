import numpy as np
import pytest

import racket_geometry


def test_inertia_tensor_principal_axes():
    """I1 < I2 < I3 and the principal axes are orthonormal, for the default
    geometry (handle_length=1, hoop_length=2, hoop_width=1, hoop_fraction=0.60)
    matching dzhanibekov_display.py's defaults."""
    result = racket_geometry.build_racket()

    I1, I2, I3 = result["I"]
    assert I1 < I2 < I3

    evecs = result["evecs"]
    assert evecs.shape == (3, 3)
    np.testing.assert_allclose(evecs.T @ evecs, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(evecs), 1.0, atol=1e-9)


def test_hoop_and_handle_indices_partition_vertices():
    result = racket_geometry.build_racket()
    n_verts = result["verts_body"].shape[0]
    all_idx = np.concatenate([result["handle_idx"], result["hoop_idx"]])
    assert sorted(all_idx.tolist()) == list(range(n_verts))


def test_axis_roles_identify_stable_and_unstable():
    result = racket_geometry.build_racket()
    assert result["imin"] != result["imid"] != result["imax"]
    assert {result["imin"], result["imid"], result["imax"]} == {0, 1, 2}


def test_face_normal_is_unit_and_aligned_with_imax_axis():
    """The racket is planar (z=0 for every vertex pre-rotation), so by the
    perpendicular axis theorem the out-of-plane normal is automatically a
    principal axis with the *largest* moment -- it must come out exactly along
    +/- the imax standard basis vector in the body frame, not merely close."""
    result = racket_geometry.build_racket()
    n = result["face_normal_body"]
    imax = result["imax"]

    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)
    expected = np.zeros(3)
    expected[imax] = 1.0
    assert (
        np.allclose(n, expected, atol=1e-9) or np.allclose(n, -expected, atol=1e-9)
    )


def test_face_normal_is_perpendicular_to_hoop_plane():
    """Sanity check from the other direction: every hoop vertex, projected onto
    the face normal, should sit at the same offset (the hoop is flat)."""
    result = racket_geometry.build_racket()
    hoop_pts = result["verts_body"][result["hoop_idx"]]
    offsets = hoop_pts @ result["face_normal_body"]
    np.testing.assert_allclose(offsets, offsets[0], atol=1e-9)
