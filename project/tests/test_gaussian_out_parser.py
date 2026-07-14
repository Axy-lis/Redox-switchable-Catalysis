from __future__ import annotations

from pathlib import Path

from src.parsers.gaussian_out_parser import (
    parse_gaussian_final_geometry,
    parse_gaussian_standard_orientation_blocks,
)
from src.pipeline.build_atom_tables import build_annotation_template, build_atom_table

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gaussian"


def test_parse_all_gaussian_orientation_blocks() -> None:
    blocks = parse_gaussian_standard_orientation_blocks(
        str(FIXTURE_DIR / "two_step_water.out")
    )
    assert len(blocks) == 2
    assert len(blocks[0].atoms) == 3
    assert blocks[0].atoms[0].element == "O"


def test_parse_final_gaussian_geometry_uses_last_block() -> None:
    molecule = parse_gaussian_final_geometry(str(FIXTURE_DIR / "two_step_water.out"))
    assert len(molecule.atoms) == 3
    assert molecule.atoms[0].coord[0] == 0.01
    assert molecule.atoms[1].element == "H"


def test_build_atom_table_from_out_file() -> None:
    df = build_atom_table(str(FIXTURE_DIR))
    assert len(df) == 3
    assert df["file_name"].iloc[0] == "two_step_water.out"
    assert df["atom_index"].tolist() == [1, 2, 3]
    assert df["element"].tolist() == ["O", "H", "H"]


def test_build_annotation_template() -> None:
    atom_table = build_atom_table(str(FIXTURE_DIR))
    template = build_annotation_template(atom_table)
    assert len(template) == 3
    assert "atom_role" in template.columns
    assert template["atom_role"].eq("").all()
