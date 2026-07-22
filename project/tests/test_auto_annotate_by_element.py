from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.auto_annotate_by_element import KEY_COLUMNS, auto_annotate_by_element


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "auto_annotate"


def _blank_annotation(atom_table: pd.DataFrame) -> pd.DataFrame:
    annotation = atom_table[KEY_COLUMNS].copy()
    annotation["atom_role"] = ""
    annotation["notes"] = ""
    return annotation


def test_auto_annotate_by_element_labels_real_catalyst_structure(tmp_path) -> None:
    atom_table = pd.read_csv(FIXTURES / "ti_salfen_ox_atom_coordinates.csv")
    annotation = _blank_annotation(atom_table)

    atom_table_csv = tmp_path / "atom_table.csv"
    annotation_csv = tmp_path / "annotation.csv"
    output_csv = tmp_path / "annotated.csv"
    atom_table.to_csv(atom_table_csv, index=False)
    annotation.to_csv(annotation_csv, index=False)

    annotated = auto_annotate_by_element(
        str(atom_table_csv), str(annotation_csv), str(output_csv)
    )

    counts = annotated["atom_role"].value_counts().to_dict()
    assert output_csv.exists()
    assert counts["metal_center"] == 1
    assert counts["ferrocene_Fe"] == 1
    assert counts["ferrocene_C"] == 10
    assert counts["donor_N"] == 2
    assert counts["phenoxy_O"] == 2
    assert counts["alkoxide_O"] == 2


def test_auto_annotate_by_element_labels_real_ground_state_monomer_structure(tmp_path) -> None:
    atom_table = pd.read_csv(FIXTURES / "ti_salfen_ox_cl_atom_coordinates.csv")
    annotation = _blank_annotation(atom_table)

    atom_table_csv = tmp_path / "atom_table.csv"
    annotation_csv = tmp_path / "annotation.csv"
    output_csv = tmp_path / "annotated.csv"
    atom_table.to_csv(atom_table_csv, index=False)
    annotation.to_csv(annotation_csv, index=False)

    annotated = auto_annotate_by_element(
        str(atom_table_csv), str(annotation_csv), str(output_csv)
    )

    counts = annotated["atom_role"].value_counts().to_dict()
    assert output_csv.exists()
    assert counts["metal_center"] == 1
    assert counts["ferrocene_Fe"] == 1
    assert counts["ferrocene_C"] == 10
    assert counts["donor_N"] == 2
    assert counts["phenoxy_O"] == 2
    assert counts["alkoxide_O"] == 2
    assert counts["monomer_C"] == 6
    assert counts["monomer_O"] == 2


def test_auto_annotate_by_element_labels_split_tmc_transition_state_consistently(tmp_path) -> None:
    atom_table = pd.read_csv(FIXTURES / "al_salfen_ox_tmc_ts1_atom_coordinates.csv")
    annotation = _blank_annotation(atom_table)

    atom_table_csv = tmp_path / "atom_table.csv"
    annotation_csv = tmp_path / "annotation.csv"
    output_csv = tmp_path / "annotated.csv"
    atom_table.to_csv(atom_table_csv, index=False)
    annotation.to_csv(annotation_csv, index=False)

    annotated = auto_annotate_by_element(
        str(atom_table_csv), str(annotation_csv), str(output_csv)
    )

    counts = annotated["atom_role"].value_counts().to_dict()
    assert output_csv.exists()
    assert counts["metal_center"] == 1
    assert counts["ferrocene_Fe"] == 1
    assert counts["ferrocene_C"] == 10
    assert counts["donor_N"] == 2
    assert counts["phenoxy_O"] == 2
    assert counts["alkoxide_O"] == 1
    assert counts["monomer_C"] == 4
    assert counts["monomer_O"] == 3
