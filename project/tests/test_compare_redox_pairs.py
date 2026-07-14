from __future__ import annotations

import pandas as pd

from src.pipeline.compare_redox_pairs import compare_redox_pairs


def test_compare_redox_pairs(tmp_path) -> None:
    selected_atoms = pd.DataFrame(
        [
            {
                "dataset_group": "catalysts",
                "file_name": "demo-ox.out",
                "structure_name": "demo-ox",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 1,
                "element": "Ti",
                "atom_role": "metal_center",
                "notes": "",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "dataset_group": "catalysts",
                "file_name": "demo-ox.out",
                "structure_name": "demo-ox",
                "metal_guess": "Ti",
                "redox_state_guess": "ox",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 2,
                "element": "N",
                "atom_role": "donor_N",
                "notes": "",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "dataset_group": "catalysts",
                "file_name": "demo-red.out",
                "structure_name": "demo-red",
                "metal_guess": "Ti",
                "redox_state_guess": "red",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 1,
                "element": "Ti",
                "atom_role": "metal_center",
                "notes": "",
                "x": 1.0,
                "y": 1.0,
                "z": 0.0,
            },
            {
                "dataset_group": "catalysts",
                "file_name": "demo-red.out",
                "structure_name": "demo-red",
                "metal_guess": "Ti",
                "redox_state_guess": "red",
                "monomer_guess": "",
                "ts_label_guess": "",
                "atom_index": 2,
                "element": "N",
                "atom_role": "donor_N",
                "notes": "",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
            },
        ]
    )

    selected_atoms_csv = tmp_path / "selected.csv"
    output_csv = tmp_path / "comparison.csv"
    selected_atoms.to_csv(selected_atoms_csv, index=False)

    comparison = compare_redox_pairs(str(selected_atoms_csv), str(output_csv))

    assert output_csv.exists()
    assert len(comparison) == 1
    assert comparison.iloc[0]["pair_key"] == "demo"
    assert comparison.iloc[0]["selected_atom_count"] == 2
    assert comparison.iloc[0]["selected_atom_rmsd"] == 0.0
    assert comparison.iloc[0]["donor_atom_rmsd"] == 0.0
    assert comparison.iloc[0]["mean_metal_to_donor_N_change"] == 0.0
    assert comparison.iloc[0]["mean_metal_to_any_donor_change"] == 0.0
