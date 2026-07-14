from __future__ import annotations

import pandas as pd

from src.pipeline.export_selected_atoms import export_selected_atoms


def test_export_selected_atoms(tmp_path) -> None:
    atom_table = pd.DataFrame(
        [
            {
                "dataset_group": "catalysts",
                "file_name": "demo.out",
                "structure_name": "demo",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 1,
                "element": "Ti",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
            },
            {
                "dataset_group": "catalysts",
                "file_name": "demo.out",
                "structure_name": "demo",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 2,
                "element": "O",
                "x": 1.1,
                "y": 1.2,
                "z": 1.3,
            },
        ]
    )
    annotation_table = pd.DataFrame(
        [
            {
                "dataset_group": "catalysts",
                "file_name": "demo.out",
                "structure_name": "demo",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 1,
                "element": "Ti",
                "atom_role": "metal_center",
                "notes": "",
            },
            {
                "dataset_group": "catalysts",
                "file_name": "demo.out",
                "structure_name": "demo",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 2,
                "element": "O",
                "atom_role": "",
                "notes": "",
            },
        ]
    )

    atom_table_csv = tmp_path / "atom_table.csv"
    annotation_csv = tmp_path / "annotation.csv"
    output_csv = tmp_path / "selected.csv"

    atom_table.to_csv(atom_table_csv, index=False)
    annotation_table.to_csv(annotation_csv, index=False)

    selected = export_selected_atoms(
        str(atom_table_csv), str(annotation_csv), str(output_csv)
    )

    assert output_csv.exists()
    assert len(selected) == 1
    assert selected.iloc[0]["atom_role"] == "metal_center"
    assert selected.iloc[0]["x"] == 0.1
