"""Build and later assemble the independently approved structural regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from regions.center_wall import (
    audit_center_wall,
    build_center_wall,
    write_gmsh as write_center_wall_gmsh,
    write_vtu as write_center_wall_vtu,
)
from regions.left_wall import (
    audit_left_wall,
    build_left_wall,
    write_gmsh as write_left_wall_gmsh,
    write_vtu as write_left_wall_vtu,
)
from regions.left_wedged_wall import (
    audit_left_wedged_wall,
    build_left_wedged_wall,
    write_gmsh as write_left_wedged_wall_gmsh,
    write_vtu as write_left_wedged_wall_vtu,
)
from regions.left_wedged_wall_transition import (
    audit_left_wedged_wall_transition,
    build_left_wedged_wall_transition,
    write_gmsh as write_left_wedged_wall_transition_gmsh,
    write_vtu as write_left_wedged_wall_transition_vtu,
)
from regions.plinth import audit_plinth, build_plinth, write_gmsh, write_vtu
from regions.right_wall import (
    audit_right_wall,
    build_right_wall,
    write_gmsh as write_right_wall_gmsh,
    write_vtu as write_right_wall_vtu,
)
from regions.right_wedged_wall import (
    audit_right_wedged_wall,
    build_right_wedged_wall,
    write_gmsh as write_right_wedged_wall_gmsh,
    write_vtu as write_right_wedged_wall_vtu,
)
from regions.right_wedged_wall_transition import (
    audit_right_wedged_wall_transition,
    build_right_wedged_wall_transition,
    write_gmsh as write_right_wedged_wall_transition_gmsh,
    write_vtu as write_right_wedged_wall_transition_vtu,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "structure_output"


def convert_with_elmergrid(mesh_path: Path, output_dir: Path) -> bool:
    executable = shutil.which("ElmerGrid")
    if executable is None:
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    subprocess.run(
        [executable, "14", "2", str(mesh_path), "-autoclean", "-out", str(output_dir)],
        check=True,
        cwd=mesh_path.parent,
    )
    return True


def build_plinth_region() -> dict[str, object]:
    mesh = build_plinth(ROOT)
    mesh_path = OUTPUT_DIR / "plinth.msh"
    preview_path = OUTPUT_DIR / "plinth.vtu"
    write_gmsh(mesh, mesh_path)
    write_vtu(mesh, preview_path)
    report = audit_plinth(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "plinth_mesh")
    return report


def build_center_wall_region() -> dict[str, object]:
    mesh = build_center_wall(ROOT)
    mesh_path = OUTPUT_DIR / "center_wall.msh"
    preview_path = OUTPUT_DIR / "center_wall.vtu"
    write_center_wall_gmsh(mesh, mesh_path)
    write_center_wall_vtu(mesh, preview_path)
    report = audit_center_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "center_wall_mesh")
    return report


def build_left_wall_region() -> dict[str, object]:
    mesh = build_left_wall(ROOT)
    mesh_path = OUTPUT_DIR / "left_wall.msh"
    preview_path = OUTPUT_DIR / "left_wall.vtu"
    write_left_wall_gmsh(mesh, mesh_path)
    write_left_wall_vtu(mesh, preview_path)
    report = audit_left_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "left_wall_mesh")
    return report


def build_left_wedged_wall_region() -> dict[str, object]:
    mesh = build_left_wedged_wall(ROOT)
    mesh_path = OUTPUT_DIR / "left_wedged_wall.msh"
    preview_path = OUTPUT_DIR / "left_wedged_wall.vtu"
    write_left_wedged_wall_gmsh(mesh, mesh_path)
    write_left_wedged_wall_vtu(mesh, preview_path)
    report = audit_left_wedged_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "left_wedged_wall_mesh")
    return report


def build_left_wedged_wall_transition_region() -> dict[str, object]:
    mesh = build_left_wedged_wall_transition(ROOT)
    mesh_path = OUTPUT_DIR / "left_wedged_wall_transition.msh"
    preview_path = OUTPUT_DIR / "left_wedged_wall_transition.vtu"
    write_left_wedged_wall_transition_gmsh(mesh, mesh_path)
    write_left_wedged_wall_transition_vtu(mesh, preview_path)
    report = audit_left_wedged_wall_transition(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(
        mesh_path, OUTPUT_DIR / "left_wedged_wall_transition_mesh"
    )
    return report


def build_right_wedged_wall_region() -> dict[str, object]:
    mesh = build_right_wedged_wall(ROOT)
    mesh_path = OUTPUT_DIR / "right_wedged_wall.msh"
    preview_path = OUTPUT_DIR / "right_wedged_wall.vtu"
    write_right_wedged_wall_gmsh(mesh, mesh_path)
    write_right_wedged_wall_vtu(mesh, preview_path)
    report = audit_right_wedged_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "right_wedged_wall_mesh")
    return report


def build_right_wedged_wall_transition_region() -> dict[str, object]:
    mesh = build_right_wedged_wall_transition(ROOT)
    mesh_path = OUTPUT_DIR / "right_wedged_wall_transition.msh"
    preview_path = OUTPUT_DIR / "right_wedged_wall_transition.vtu"
    write_right_wedged_wall_transition_gmsh(mesh, mesh_path)
    write_right_wedged_wall_transition_vtu(mesh, preview_path)
    report = audit_right_wedged_wall_transition(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(
        mesh_path, OUTPUT_DIR / "right_wedged_wall_transition_mesh"
    )
    return report


def build_right_wall_region() -> dict[str, object]:
    mesh = build_right_wall(ROOT)
    mesh_path = OUTPUT_DIR / "right_wall.msh"
    preview_path = OUTPUT_DIR / "right_wall.vtu"
    write_right_wall_gmsh(mesh, mesh_path)
    write_right_wall_vtu(mesh, preview_path)
    report = audit_right_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "right_wall_mesh")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "region",
        nargs="?",
        default="plinth",
        choices=(
            "plinth", "center_wall", "left_wall", "left_wedged_wall", "left_wedged_wall_transition",
            "right_wedged_wall", "right_wedged_wall_transition", "right_wall",
        ),
    )
    args = parser.parse_args()
    if args.region == "plinth":
        print(json.dumps(build_plinth_region(), indent=2))
    elif args.region == "center_wall":
        print(json.dumps(build_center_wall_region(), indent=2))
    elif args.region == "left_wall":
        print(json.dumps(build_left_wall_region(), indent=2))
    elif args.region == "left_wedged_wall":
        print(json.dumps(build_left_wedged_wall_region(), indent=2))
    elif args.region == "left_wedged_wall_transition":
        print(json.dumps(build_left_wedged_wall_transition_region(), indent=2))
    elif args.region == "right_wedged_wall":
        print(json.dumps(build_right_wedged_wall_region(), indent=2))
    elif args.region == "right_wedged_wall_transition":
        print(json.dumps(build_right_wedged_wall_transition_region(), indent=2))
    elif args.region == "right_wall":
        print(json.dumps(build_right_wall_region(), indent=2))


if __name__ == "__main__":
    main()
