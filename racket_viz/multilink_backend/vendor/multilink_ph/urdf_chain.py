from __future__ import annotations

from pathlib import Path


def write_planar_chain_urdf(
    path: str | Path,
    n_links: int = 6,
    total_length: float = 1.0,
    total_mass: float = 0.3,
    radius: float = 0.015,
    include_contact_frames: bool = True,
) -> Path:
    """Write a minimal serial-chain URDF suitable for a Pinocchio planar root.

    The URDF itself has no floating joint; load it with `JointModelPlanar()`.
    Every link frame can be used as an anisotropic friction/contact frame.
    """
    if n_links < 2:
        raise ValueError("n_links must be >=2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    L = total_length / n_links
    m = total_mass / n_links
    Izz = m * L * L / 12.0
    Ixx = Iyy = max(1e-8, m * radius * radius / 4.0)
    lines = ['<?xml version="1.0"?>', '<robot name="multilink_planar_chain">']
    for i in range(n_links):
        lines += [
            f'  <link name="link_{i}">',
            f'    <inertial><origin xyz="{L/2:.12g} 0 0" rpy="0 0 0"/>',
            f'      <mass value="{m:.12g}"/>',
            f'      <inertia ixx="{Ixx:.12g}" ixy="0" ixz="0" iyy="{Iyy:.12g}" iyz="0" izz="{Izz:.12g}"/>',
            '    </inertial>',
            f'    <visual><origin xyz="{L/2:.12g} 0 0" rpy="0 0 0"/><geometry><box size="{L:.12g} {2*radius:.12g} {2*radius:.12g}"/></geometry></visual>',
            f'    <collision><origin xyz="{L/2:.12g} 0 0" rpy="0 0 0"/><geometry><box size="{L:.12g} {2*radius:.12g} {2*radius:.12g}"/></geometry></collision>',
            '  </link>',
        ]
    if include_contact_frames:
        for i in range(n_links):
            lines += [
                f'  <link name="contact_{i}"/>',
                f'  <joint name="contact_fixed_{i}" type="fixed">',
                f'    <parent link="link_{i}"/><child link="contact_{i}"/>',
                f'    <origin xyz="{L/2:.12g} 0 0" rpy="0 0 0"/>',
                '  </joint>',
            ]
    for i in range(1, n_links):
        lines += [
            f'  <joint name="joint_{i-1}" type="revolute">',
            f'    <parent link="link_{i-1}"/><child link="link_{i}"/>',
            f'    <origin xyz="{L:.12g} 0 0" rpy="0 0 0"/>',
            '    <axis xyz="0 0 1"/>',
            '    <limit lower="-1.5" upper="1.5" effort="20" velocity="20"/>',
            '    <dynamics damping="0" friction="0"/>',
            '  </joint>',
        ]
    lines.append('</robot>')
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
