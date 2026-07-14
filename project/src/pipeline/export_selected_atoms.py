from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_selected_atoms(
    atom_table_csv: str, annotation_csv: str, output_csv: str
) -> pd.DataFrame:
    atom_table = pd.read_csv(atom_table_csv)
    annotation_table = pd.read_csv(annotation_csv)

    annotation_table["atom_role"] = (
        annotation_table["atom_role"].fillna("").astype(str).str.strip()
    )
    annotation_table["notes"] = (
        annotation_table["notes"].fillna("").astype(str).str.strip()
    )

    selected = annotation_table[annotation_table["atom_role"] != ""].copy()
    if selected.empty:
        raise ValueError(f"No annotated atoms found in: {annotation_csv}")

    merged = selected.merge(
        atom_table,
        on=[
            "dataset_group",
            "file_name",
            "structure_name",
            "metal_guess",
            "redox_state_guess",
            "monomer_guess",
            "ts_label_guess",
            "atom_index",
            "element",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_coords = merged[["x", "y", "z"]].isna().any(axis=1)
    if missing_coords.any():
        missing_rows = merged.loc[missing_coords, ["file_name", "atom_index", "element"]]
        raise ValueError(
            "Some annotated atoms were not found in the atom table: "
            f"{missing_rows.to_dict(orient='records')}"
        )

    merged = merged[
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
            "atom_role",
            "notes",
            "x",
            "y",
            "z",
        ]
    ].sort_values(
        by=["file_name", "atom_role", "atom_index"], kind="stable"
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a coordinate table containing only manually annotated atoms."
    )
    parser.add_argument("atom_table_csv", help="Full atom coordinate table CSV")
    parser.add_argument("annotation_csv", help="Annotation CSV with atom_role values")
    parser.add_argument("output_csv", help="Output CSV for selected atoms only")
    args = parser.parse_args()

    selected = export_selected_atoms(
        args.atom_table_csv, args.annotation_csv, args.output_csv
    )
    print(f"Wrote {len(selected)} selected atom rows to {args.output_csv}")


if __name__ == "__main__":
    main()
