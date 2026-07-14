from __future__ import annotations

from pathlib import Path

import numpy as np

from src.schema import Atom, Molecule


def parse_xyz(file_path: str) -> Molecule:
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"XYZ file does not exist: {file_path}")

    raw_lines = path.read_text().splitlines()
    if len(raw_lines) < 2:
        raise ValueError(f"Malformed XYZ file (too few lines): {file_path}")

    try:
        atom_count = int(raw_lines[0].strip())
        atom_count_line_index = 0
    except ValueError as exc:
        atom_count = None
        atom_count_line_index = -1
        for i, line in enumerate(raw_lines):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                atom_count = int(candidate)
                atom_count_line_index = i
                break
            except ValueError:
                continue
        if atom_count is None or atom_count_line_index < 0:
            raise ValueError(
                f"Malformed XYZ file (invalid atom count): {file_path}"
            ) from exc

    atoms: list[Atom] = []
    atom_start_index = atom_count_line_index + 2

    i = atom_start_index
    parsed = 0
    while parsed < atom_count:
        if i >= len(raw_lines):
            raise ValueError(
                f"Malformed XYZ file (expected {atom_count} atoms, found {parsed}): {file_path}"
            )
        line = raw_lines[i].strip()
        i += 1
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line {i}: {line}")
        element = parts[0]
        try:
            coord = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=float
            )
        except ValueError as exc:
            raise ValueError(f"Malformed XYZ coordinates on line {i}: {line}") from exc

        atoms.append(Atom(element=element, coord=coord))
        parsed += 1

    return Molecule(atoms=atoms)
