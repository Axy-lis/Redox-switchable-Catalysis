from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.parsers.gaussian_out_parser import parse_gaussian_final_geometry
from src.parsers.xyz_parser import parse_xyz


def _parse_structure(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".xyz":
        return parse_xyz(str(file_path))
    if suffix == ".out":
        return parse_gaussian_final_geometry(str(file_path))
    raise ValueError(f"Unsupported structure format: {file_path}")


def _infer_metadata(file_path: Path, input_root: Path) -> dict[str, Any]:
    stem_parts = file_path.stem.split("-")
    return {
        "dataset_group": file_path.parent.name,
        "relative_path": str(file_path.relative_to(input_root)),
        "file_name": file_path.name,
        "structure_name": file_path.stem,
        "metal_guess": stem_parts[0] if stem_parts else "",
        "redox_state_guess": "ox" if "ox" in stem_parts else ("red" if "red" in stem_parts else ""),
        "monomer_guess": next(
            (part for part in stem_parts if part in {"CL", "LA", "VL", "TMC"}),
            "",
        ),
        "ts_label_guess": next((part for part in stem_parts if part.startswith("TS")), ""),
    }


def build_atom_table(input_folder: str) -> pd.DataFrame:
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    structure_files = sorted(
        [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".xyz", ".out"}
        ]
    )
    if not structure_files:
        raise ValueError(f"No .xyz or .out files found in: {input_folder}")

    rows: list[dict[str, Any]] = []
    for structure_file in structure_files:
        metadata = _infer_metadata(structure_file, input_path)
        molecule = _parse_structure(structure_file)
        for atom_index, atom in enumerate(molecule.atoms, start=1):
            rows.append(
                {
                    **metadata,
                    "atom_index": atom_index,
                    "element": atom.element,
                    "x": float(atom.coord[0]),
                    "y": float(atom.coord[1]),
                    "z": float(atom.coord[2]),
                }
            )

    return pd.DataFrame(rows)


def build_annotation_template(atom_table: pd.DataFrame) -> pd.DataFrame:
    template = atom_table[
        [
            "dataset_group",
            "file_name",
            "structure_name",
            "metal_guess",
            "redox_state_guess",
            "monomer_guess",
            "ts_label_guess",
            "atom_index",
            "element",
        ]
    ].copy()
    template["atom_role"] = ""
    template["notes"] = ""
    return template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build atom coordinate and annotation tables from .xyz/.out files."
    )
    parser.add_argument("input_folder", help="Folder containing structure files")
    parser.add_argument("atom_table_csv", help="Output CSV for atom coordinates")
    parser.add_argument(
        "annotation_template_csv",
        help="Output CSV for manual atom-role annotation",
    )
    args = parser.parse_args()

    atom_table = build_atom_table(args.input_folder)
    annotation_template = build_annotation_template(atom_table)

    atom_table_path = Path(args.atom_table_csv)
    annotation_path = Path(args.annotation_template_csv)
    atom_table_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    atom_table.to_csv(atom_table_path, index=False)
    annotation_template.to_csv(annotation_path, index=False)
    print(
        f"Wrote {len(atom_table)} atom rows to {atom_table_path} and "
        f"{len(annotation_template)} annotation rows to {annotation_path}"
    )


if __name__ == "__main__":
    main()
