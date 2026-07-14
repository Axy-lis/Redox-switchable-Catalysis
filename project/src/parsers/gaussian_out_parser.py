from __future__ import annotations

from pathlib import Path

import numpy as np

from src.schema import Atom, Molecule

_PERIODIC_TABLE = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]


def _is_dash_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


def _atomic_number_to_symbol(atomic_number: int) -> str:
    if atomic_number <= 0 or atomic_number >= len(_PERIODIC_TABLE):
        raise ValueError(f"Unsupported atomic number in Gaussian output: {atomic_number}")
    return _PERIODIC_TABLE[atomic_number]


def parse_gaussian_standard_orientation_blocks(file_path: str) -> list[Molecule]:
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Gaussian output file does not exist: {file_path}")

    raw_lines = path.read_text(errors="replace").splitlines()
    blocks: list[Molecule] = []

    i = 0
    while i < len(raw_lines):
        if "Standard orientation:" not in raw_lines[i]:
            i += 1
            continue

        j = i + 1
        while j < len(raw_lines) and not _is_dash_line(raw_lines[j]):
            j += 1
        if j >= len(raw_lines):
            break

        j += 1
        while j < len(raw_lines) and not _is_dash_line(raw_lines[j]):
            j += 1
        if j >= len(raw_lines):
            break

        data_start = j + 1
        atoms: list[Atom] = []
        k = data_start
        while k < len(raw_lines) and not _is_dash_line(raw_lines[k]):
            parts = raw_lines[k].split()
            if len(parts) < 6:
                raise ValueError(
                    f"Malformed Gaussian coordinate line in {file_path}: {raw_lines[k]}"
                )
            atomic_number = int(parts[1])
            element = _atomic_number_to_symbol(atomic_number)
            coord = np.array(
                [float(parts[3]), float(parts[4]), float(parts[5])], dtype=float
            )
            atoms.append(Atom(element=element, coord=coord))
            k += 1

        if atoms:
            blocks.append(Molecule(atoms=atoms))

        i = k

    if not blocks:
        raise ValueError(
            f"No Gaussian 'Standard orientation' blocks found in: {file_path}"
        )

    return blocks


def parse_gaussian_final_geometry(file_path: str) -> Molecule:
    return parse_gaussian_standard_orientation_blocks(file_path)[-1]
