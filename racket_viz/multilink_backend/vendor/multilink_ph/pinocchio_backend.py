from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np


class PinocchioUnavailableError(RuntimeError):
    pass


def _import_pinocchio():
    try:
        import pinocchio as pin  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise PinocchioUnavailableError(
            "Pinocchio is not installed. Install the Python bindings with either "
            "`conda install pinocchio -c conda-forge` or, on supported Linux systems, "
            "`pip install pin`."
        ) from exc
    return pin


@dataclass(frozen=True)
class RigidWorldContactSpec:
    """Persistent rigid contact between a robot frame and the world.

    This is appropriate for a *known active contact mode*.  It is deliberately
    separate from unilateral contact detection: activation/deactivation is a
    hybrid event and should not be hidden inside the pH damping matrix.
    """

    frame_name: str
    contact_type: str = "3D"  # "3D" locks translation, "6D" locks full pose.


@dataclass(frozen=True)
class FrameFrictionSpec:
    """Smooth anisotropic friction acting at a named frame.

    This gives a differentiability-friendly contact port for the first
    Pinocchio experiment.  It is useful for wheel/scale-like snake contacts,
    where rigid CONTACT_3D would over-constrain tangential motion.
    """

    frame_name: str
    c_tangent: float = 0.25
    c_normal: float = 1.0
    weight: float = 1.0
    regularization: float = 1e-6


class PinocchioRigidBodyBackend:
    """Thin NumPy wrapper around Pinocchio for the current pH codebase.

    The wrapper exposes three groups of operations:

    1. manifold-aware configuration operations (`integrate`, `difference`),
    2. rigid-body dynamics and analytical derivatives (ABA/RNEA-level terms),
    3. rigid fixed-mode contact dynamics and contact-force derivatives.

    Configuration derivatives returned by Pinocchio live in the tangent space
    (dimension `model.nv`) even when `model.nq != model.nv`.
    """

    def __init__(self, model, root_joint: str = "fixed") -> None:
        self.pin = _import_pinocchio()
        self.model = model
        self.data = model.createData()
        self.root_joint = root_joint
        if root_joint == "planar":
            self.base_nv = 3
            self.base_nq = 4
        elif root_joint == "freeflyer":
            self.base_nv = 6
            self.base_nq = 7
        elif root_joint == "fixed":
            self.base_nv = 0
            self.base_nq = 0
        else:
            raise ValueError(f"unsupported root_joint={root_joint!r}")
        self._rigid_contact_models = []
        self._rigid_contact_datas = []
        self._rigid_contact_specs: list[RigidWorldContactSpec] = []
        self._planar_contact_specs: list[RigidWorldContactSpec] = []
        self._planar_contact_anchors: list[np.ndarray] = []

    @classmethod
    def from_urdf(cls, urdf: str | Path, root_joint: str = "planar") -> "PinocchioRigidBodyBackend":
        pin = _import_pinocchio()
        urdf = str(Path(urdf))
        if root_joint == "planar":
            model = pin.buildModelFromUrdf(urdf, pin.JointModelPlanar())
        elif root_joint == "freeflyer":
            model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
        elif root_joint == "fixed":
            model = pin.buildModelFromUrdf(urdf)
        else:
            raise ValueError("root_joint must be one of: planar, freeflyer, fixed")
        return cls(model, root_joint=root_joint)

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def n_actuated(self) -> int:
        return self.nv - self.base_nv

    def neutral(self) -> np.ndarray:
        return np.asarray(self.pin.neutral(self.model), dtype=float).copy()

    def integrate(self, q: np.ndarray, tangent_increment: np.ndarray) -> np.ndarray:
        return np.asarray(self.pin.integrate(self.model, q, tangent_increment), dtype=float).copy()

    def difference(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        return np.asarray(self.pin.difference(self.model, q0, q1), dtype=float).copy()

    def planar_configuration(self, pose_xytheta: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """Construct native Pinocchio q for the included planar-chain URDF.

        Pinocchio's planar root uses [x, y, cos(theta), sin(theta)] while the
        tangent/root velocity has dimension three.  This helper is intentionally
        limited to the generated chain, whose remaining joints are scalar.
        """
        if self.root_joint != "planar":
            raise ValueError("planar_configuration requires root_joint='planar'")
        pose = np.asarray(pose_xytheta, dtype=float)
        joints = np.asarray(joints, dtype=float)
        if pose.shape != (3,):
            raise ValueError("pose_xytheta must have shape (3,)")
        if joints.shape != (self.n_actuated,):
            raise ValueError(f"expected {self.n_actuated} joints, got {joints.shape}")
        q = self.neutral()
        q[:4] = [pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2])]
        q[4:] = joints
        return q

    def planar_velocity(self, body_twist: np.ndarray, joint_velocity: np.ndarray) -> np.ndarray:
        if self.root_joint != "planar":
            raise ValueError("planar_velocity requires root_joint='planar'")
        xi = np.asarray(body_twist, dtype=float)
        qd = np.asarray(joint_velocity, dtype=float)
        if xi.shape != (3,) or qd.shape != (self.n_actuated,):
            raise ValueError("bad planar velocity dimensions")
        return np.concatenate([xi, qd])

    def actuation_vector(self, joint_torque: np.ndarray) -> np.ndarray:
        u = np.asarray(joint_torque, dtype=float)
        if u.shape != (self.n_actuated,):
            raise ValueError(f"expected joint torque shape {(self.n_actuated,)}, got {u.shape}")
        tau = np.zeros(self.nv, dtype=float)
        tau[self.base_nv :] = u
        return tau

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        M_upper = np.asarray(self.pin.crba(self.model, self.data, q), dtype=float).copy()
        # CRBA stores the relevant symmetric part in the upper triangle.
        return np.triu(M_upper) + np.triu(M_upper, 1).T

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(self.pin.nonLinearEffects(self.model, self.data, q, v), dtype=float).copy()

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q), dtype=float).copy()

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, derivatives: bool = False) -> dict[str, np.ndarray]:
        ddq = np.asarray(self.pin.aba(self.model, self.data, q, v, tau), dtype=float).copy()
        out = {"ddq": ddq}
        if derivatives:
            # These are analytical algorithmic derivatives in tangent coordinates.
            self.pin.computeABADerivatives(self.model, self.data, q, v, tau)
            out.update(
                ddq_dq=np.asarray(self.data.ddq_dq, dtype=float).copy(),
                ddq_dv=np.asarray(self.data.ddq_dv, dtype=float).copy(),
                ddq_dtau=np.asarray(self.data.ddq_dtau, dtype=float).copy(),
            )
        return out

    def rnea_derivatives(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
        self.pin.computeRNEADerivatives(self.model, self.data, q, v, a)
        return {
            "dtau_dq": np.asarray(self.data.dtau_dq, dtype=float).copy(),
            "dtau_dv": np.asarray(self.data.dtau_dv, dtype=float).copy(),
            # For standard rigid-body mechanics dtau/da is the mass matrix.
            "dtau_da": self.mass_matrix(q),
        }

    def _frame_id(self, frame_name: str) -> int:
        idx = int(self.model.getFrameId(frame_name))
        if idx >= int(self.model.nframes):
            raise KeyError(f"frame {frame_name!r} not present in model")
        return idx

    def frame_placement(self, q: np.ndarray, frame_name: str):
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self._frame_id(frame_name)].copy()

    def frame_jacobian(self, q: np.ndarray, frame_name: str, reference: str = "LOCAL_WORLD_ALIGNED") -> np.ndarray:
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        rf = getattr(self.pin.ReferenceFrame, reference)
        return np.asarray(
            self.pin.getFrameJacobian(self.model, self.data, self._frame_id(frame_name), rf),
            dtype=float,
        ).copy()

    def frame_jacobian_time_variation(
        self, q: np.ndarray, v: np.ndarray, frame_name: str, reference: str = "LOCAL_WORLD_ALIGNED"
    ) -> np.ndarray:
        rf = getattr(self.pin.ReferenceFrame, reference)
        return np.asarray(
            self.pin.frameJacobianTimeVariation(
                self.model, self.data, q, v, self._frame_id(frame_name), rf
            ),
            dtype=float,
        ).copy()

    def frame_velocity(self, q: np.ndarray, v: np.ndarray, frame_name: str, reference: str = "LOCAL_WORLD_ALIGNED") -> np.ndarray:
        self.pin.forwardKinematics(self.model, self.data, q, v)
        self.pin.updateFramePlacements(self.model, self.data)
        rf = getattr(self.pin.ReferenceFrame, reference)
        motion = self.pin.getFrameVelocity(self.model, self.data, self._frame_id(frame_name), rf)
        return np.asarray(motion.vector, dtype=float).copy()

    def anisotropic_frame_friction(
        self,
        q: np.ndarray,
        v: np.ndarray,
        specs: Sequence[FrameFrictionSpec],
    ) -> dict[str, np.ndarray | float]:
        """Generalized smooth friction force and dissipated power.

        Pinocchio serializes spatial motion as [linear(3), angular(3)].  The
        first three rows of a LOCAL_WORLD_ALIGNED frame Jacobian therefore map
        generalized velocity to point linear velocity.  Tangential and lateral
        axes are taken from the frame's x/y axes in world coordinates.
        """
        tau_contact = np.zeros(self.nv, dtype=float)
        power = 0.0
        forces = []
        for spec in specs:
            placement = self.frame_placement(q, spec.frame_name)
            J = self.frame_jacobian(q, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            Jlin = J[:3, :]
            vel = Jlin @ v
            t = np.asarray(placement.rotation[:, 0], dtype=float)
            n = np.asarray(placement.rotation[:, 1], dtype=float)
            vt = float(np.dot(vel, t))
            vn = float(np.dot(vel, n))
            # Linear Rayleigh law.  It is intentionally smooth so the first
            # contact experiment isolates mechanics from hybrid mode switching.
            f = -spec.weight * (spec.c_tangent * vt * t + spec.c_normal * vn * n)
            tau_contact += Jlin.T @ f
            p = float(np.dot(f, vel))
            power += p
            forces.append(f)
        lateral_speeds = []
        tangential_speeds = []
        for spec in specs:
            placement = self.frame_placement(q, spec.frame_name)
            J = self.frame_jacobian(q, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            vel = J[:3, :] @ v
            t = np.asarray(placement.rotation[:, 0], dtype=float)
            n = np.asarray(placement.rotation[:, 1], dtype=float)
            tangential_speeds.append(abs(float(np.dot(vel, t))))
            lateral_speeds.append(abs(float(np.dot(vel, n))))
        return {
            "tau_contact": tau_contact,
            "contact_power": power,
            "forces": np.asarray(forces, dtype=float) if forces else np.zeros((0, 3)),
            "max_lateral_speed": max(lateral_speeds, default=0.0),
            "max_tangential_speed": max(tangential_speeds, default=0.0),
        }

    def kinetic_energy(self, q: np.ndarray, v: np.ndarray) -> float:
        """Return 1/2 v^T M(q) v using CRBA's inertia matrix."""
        M = self.mass_matrix(q)
        return 0.5 * float(np.dot(v, M @ v))

    def planar_centroidal_momentum(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return planar centroidal momentum [Lz, Px, Py].

        Pinocchio computes the centroidal spatial momentum from the full
        articulated model.  This is the preferred global momentum map monitor
        for the Souriau--Fisher diagnostic; it avoids assuming that arbitrary
        generalized momentum coordinates equal physical total momentum.
        """
        if hasattr(self.pin, "computeCentroidalMomentum"):
            self.pin.computeCentroidalMomentum(self.model, self.data, q, v)
        else:  # compatibility with older bindings
            self.pin.ccrba(self.model, self.data, q, v)
        hg = self.data.hg
        lin = np.asarray(hg.linear, dtype=float)
        ang = np.asarray(hg.angular, dtype=float)
        return np.asarray([ang[2], lin[0], lin[1]], dtype=float)

    def planar_pose(self, q: np.ndarray) -> np.ndarray:
        """Extract [x,y,theta] from Pinocchio's planar-root configuration."""
        if self.root_joint != "planar":
            raise ValueError("planar_pose requires root_joint='planar'")
        return np.asarray([q[0], q[1], np.arctan2(q[3], q[2])], dtype=float)

    def joint_configuration(self, q: np.ndarray) -> np.ndarray:
        """Internal scalar joint coordinates for the generated chain."""
        return np.asarray(q[self.base_nq :], dtype=float).copy()

    def frame_world_position(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        placement = self.frame_placement(q, frame_name)
        return np.asarray(placement.translation, dtype=float).copy()

    def rigid_contact_jacobian(
        self, q: np.ndarray, specs: Sequence[RigidWorldContactSpec]
    ) -> np.ndarray:
        """Stack active contact Jacobians in LOCAL_WORLD_ALIGNED coordinates."""
        blocks = []
        for spec in specs:
            J = self.frame_jacobian(q, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            blocks.append(J[:3, :] if spec.contact_type.upper() == "3D" else J)
        if not blocks:
            return np.zeros((0, self.nv), dtype=float)
        return np.vstack(blocks)

    def project_velocity_to_rigid_contacts(
        self,
        q: np.ndarray,
        v: np.ndarray,
        specs: Sequence[RigidWorldContactSpec],
        regularization: float = 1e-10,
    ) -> np.ndarray:
        """Mass-metric projection onto J_c v = 0 at contact activation.

        This is the perfectly-plastic velocity projection associated with the
        active bilateral constraint and avoids starting ``constraintDynamics``
        from a kinematically inconsistent contact velocity.
        """
        J = self.rigid_contact_jacobian(q, specs)
        if J.shape[0] == 0:
            return np.asarray(v, dtype=float).copy()
        M = self.mass_matrix(q)
        Minv_JT = np.linalg.solve(M, J.T)
        W = J @ Minv_JT + regularization * np.eye(J.shape[0])
        correction = Minv_JT @ np.linalg.solve(W, J @ v)
        return np.asarray(v - correction, dtype=float)

    def active_rigid_contact_velocity_norm(self, q: np.ndarray, v: np.ndarray) -> float:
        if not self._rigid_contact_specs:
            return 0.0
        J = self.rigid_contact_jacobian(q, self._rigid_contact_specs)
        return float(np.linalg.norm(J @ v))

    def activate_planar_world_contacts(
        self, q: np.ndarray, specs: Sequence[RigidWorldContactSpec]
    ) -> None:
        """Activate x/y point constraints at each frame's current world position.

        ``JointModelPlanar`` has no z velocity, so a native CONTACT_3D contains a
        structurally redundant row.  This 2D KKT path is the default rigid-contact
        experiment for the planar chain while still using Pinocchio CRBA, ABA,
        frame Jacobians and Jacobian time variation.
        """
        self._planar_contact_specs = list(specs)
        self._planar_contact_anchors = [self.frame_world_position(q, s.frame_name)[:2] for s in specs]

    def clear_planar_world_contacts(self) -> None:
        self._planar_contact_specs = []
        self._planar_contact_anchors = []

    def planar_rigid_contact_jacobian(
        self, q: np.ndarray, specs: Sequence[RigidWorldContactSpec] | None = None
    ) -> np.ndarray:
        specs = self._planar_contact_specs if specs is None else list(specs)
        blocks = []
        for spec in specs:
            J = self.frame_jacobian(q, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            blocks.append(J[:2, :])
        return np.vstack(blocks) if blocks else np.zeros((0, self.nv), dtype=float)

    def project_velocity_to_planar_contacts(
        self, q: np.ndarray, v: np.ndarray, specs: Sequence[RigidWorldContactSpec], regularization: float = 1e-10
    ) -> np.ndarray:
        J = self.planar_rigid_contact_jacobian(q, specs)
        if J.shape[0] == 0:
            return np.asarray(v, dtype=float).copy()
        M = self.mass_matrix(q)
        Minv_JT = np.linalg.solve(M, J.T)
        W = J @ Minv_JT + regularization * np.eye(J.shape[0])
        return np.asarray(v - Minv_JT @ np.linalg.solve(W, J @ v), dtype=float)

    def planar_constraint_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        kp: float = 80.0,
        kd: float = 18.0,
        regularization: float = 1e-9,
    ) -> dict[str, np.ndarray | float]:
        if not self._planar_contact_specs:
            raise RuntimeError("no active planar contacts; call activate_planar_world_contacts first")
        free = self.aba(q, v, tau, derivatives=False)["ddq"]
        M = self.mass_matrix(q)
        J_blocks = []
        gamma_blocks = []
        desired_blocks = []
        for spec, anchor in zip(self._planar_contact_specs, self._planar_contact_anchors):
            J6 = self.frame_jacobian(q, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            dJ6 = self.frame_jacobian_time_variation(q, v, spec.frame_name, "LOCAL_WORLD_ALIGNED")
            J = J6[:2, :]
            dJ = dJ6[:2, :]
            pos = self.frame_world_position(q, spec.frame_name)[:2]
            vel = J @ v
            error = pos - anchor
            desired = -kp * error - kd * vel
            J_blocks.append(J)
            gamma_blocks.append(dJ @ v)
            desired_blocks.append(desired)
        Jc = np.vstack(J_blocks)
        gamma = np.concatenate(gamma_blocks)
        desired = np.concatenate(desired_blocks)
        Minv_JT = np.linalg.solve(M, Jc.T)
        W = Jc @ Minv_JT + regularization * np.eye(Jc.shape[0])
        # M a + h = tau + J^T lambda, J a + gamma = desired.
        lam = np.linalg.solve(W, desired - gamma - Jc @ free)
        ddq = free + Minv_JT @ lam
        contact_velocity = Jc @ v
        return {
            "ddq": np.asarray(ddq, dtype=float),
            "lambda": np.asarray(lam, dtype=float),
            "contact_power": float(np.dot(lam, contact_velocity)),
            "contact_velocity_norm": float(np.linalg.norm(contact_velocity)),
            "Jc": Jc,
        }

    def activate_rigid_world_contacts(
        self,
        q: np.ndarray,
        specs: Sequence[RigidWorldContactSpec],
    ) -> None:
        """Freeze selected frames to their *current* world placements.

        This sets a known active contact mode.  For walking/impact studies a
        higher-level hybrid contact manager should decide when this method is
        called; mode selection itself is non-smooth.
        """
        pin = self.pin
        self.pin.framesForwardKinematics(self.model, self.data, q)
        models = []
        datas = []
        for spec in specs:
            fid = self._frame_id(spec.frame_name)
            frame = self.model.frames[fid]
            ctype = pin.ContactType.CONTACT_3D if spec.contact_type.upper() == "3D" else pin.ContactType.CONTACT_6D
            local_placement = frame.placement
            world_anchor = self.data.oMf[fid].copy()
            parent_joint = int(frame.parentJoint)
            cm = pin.RigidConstraintModel(
                ctype,
                self.model,
                parent_joint,
                local_placement,
                0,
                world_anchor,
            )
            models.append(cm)
            datas.append(cm.createData())
        pin.initConstraintDynamics(self.model, self.data, models)
        self._rigid_contact_models = models
        self._rigid_contact_datas = datas
        self._rigid_contact_specs = list(specs)

    def clear_rigid_contacts(self) -> None:
        self._rigid_contact_models = []
        self._rigid_contact_datas = []
        self._rigid_contact_specs = []

    def constraint_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        derivatives: bool = False,
    ) -> dict[str, np.ndarray]:
        if not self._rigid_contact_models:
            raise RuntimeError("no active rigid contacts; call activate_rigid_world_contacts first")
        ddq = np.asarray(
            self.pin.constraintDynamics(
                self.model,
                self.data,
                q,
                v,
                tau,
                self._rigid_contact_models,
                self._rigid_contact_datas,
            ),
            dtype=float,
        ).copy()
        out = {"ddq": ddq}
        # lambda_c is the canonical storage in the RigidConstraintModel API.
        if hasattr(self.data, "lambda_c"):
            out["lambda"] = np.asarray(self.data.lambda_c, dtype=float).copy()
        if derivatives:
            self.pin.computeConstraintDynamicsDerivatives(
                self.model,
                self.data,
                self._rigid_contact_models,
                self._rigid_contact_datas,
            )
            out.update(
                ddq_dq=np.asarray(self.data.ddq_dq, dtype=float).copy(),
                ddq_dv=np.asarray(self.data.ddq_dv, dtype=float).copy(),
                ddq_dtau=np.asarray(self.data.ddq_dtau, dtype=float).copy(),
                dlambda_dq=np.asarray(self.data.dlambda_dq, dtype=float).copy(),
                dlambda_dv=np.asarray(self.data.dlambda_dv, dtype=float).copy(),
                dlambda_dtau=np.asarray(self.data.dlambda_dtau, dtype=float).copy(),
            )
        return out

    def step_semi_implicit(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        dt: float,
        friction_specs: Sequence[FrameFrictionSpec] = (),
        rigid_contact: bool = False,
        rigid_contact_solver: str = "native",
    ) -> dict[str, np.ndarray | float]:
        """pH-compatible mechanical step in generalized velocity coordinates."""
        tau_total = np.asarray(tau, dtype=float).copy()
        contact_power = 0.0
        if friction_specs:
            c = self.anisotropic_frame_friction(q, v, friction_specs)
            tau_total = tau_total + c["tau_contact"]
            contact_power = float(c["contact_power"])
        if rigid_contact:
            if rigid_contact_solver == "native":
                dyn = self.constraint_dynamics(q, v, tau_total, derivatives=False)
            elif rigid_contact_solver == "planar_kkt":
                dyn = self.planar_constraint_dynamics(q, v, tau_total)
                contact_power += float(dyn.get("contact_power", 0.0))
            else:
                raise ValueError("rigid_contact_solver must be 'native' or 'planar_kkt'")
        else:
            dyn = self.aba(q, v, tau_total, derivatives=False)
        ddq = np.asarray(dyn["ddq"], dtype=float)
        v_next = v + dt * ddq
        q_next = self.integrate(q, dt * v_next)
        out: dict[str, np.ndarray | float] = {
            "q": q_next,
            "v": v_next,
            "ddq": ddq,
            "tau_total": tau_total,
            "contact_power": contact_power,
        }
        if "lambda" in dyn:
            out["lambda"] = dyn["lambda"]
        if "contact_velocity_norm" in dyn:
            out["contact_velocity_norm"] = float(dyn["contact_velocity_norm"])
        return out
