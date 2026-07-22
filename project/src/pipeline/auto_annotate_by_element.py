from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = [
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

DONOR_ROLE_BY_ELEMENT = {
    "N": "donor_N",
    "S": "donor_S",
    "P": "donor_P",
}

AUTO_GENERATED_ROLES = {
    "alkoxide_O",
    "donor_N",
    "donor_P",
    "donor_S",
    "ferrocene_C",
    "ferrocene_Fe",
    "metal_center",
    "monomer_C",
    "monomer_N",
    "monomer_O",
    "monomer_P",
    "monomer_S",
    "phenoxy_O",
    "relevant_O",
}

_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "Fe": 1.24,
    "Al": 1.21,
    "Ti": 1.60,
    "Y": 1.90,
    "Zr": 1.75,
}

_BOND_FUDGE_FACTOR = 1.25
_EXPECTED_MONOMER_COMPOSITION = {
    "CL": {"C": 6, "O": 2},
    "LA": {"C": 6, "O": 4},
    "TMC": {"C": 4, "O": 3},
    "VL": {"C": 5, "O": 2},
}


def _clear_auto_generated_roles(annotation: pd.DataFrame) -> pd.DataFrame:
    annotation = annotation.copy()
    annotation["atom_role"] = annotation["atom_role"].fillna("").astype(str).str.strip()
    auto_mask = annotation["atom_role"].isin(AUTO_GENERATED_ROLES)
    annotation.loc[auto_mask, "atom_role"] = ""
    return annotation


def _build_adjacency(structure_df: pd.DataFrame) -> dict[int, set[int]]:
    coords = structure_df[["x", "y", "z"]].to_numpy(dtype=float)
    adjacency = {i: set() for i in range(len(structure_df))}
    for i in range(len(structure_df)):
        element_i = str(structure_df.loc[i, "element"])
        radius_i = _COVALENT_RADII.get(element_i, 0.80)
        for j in range(i + 1, len(structure_df)):
            element_j = str(structure_df.loc[j, "element"])
            radius_j = _COVALENT_RADII.get(element_j, 0.80)
            cutoff = _BOND_FUDGE_FACTOR * (radius_i + radius_j)
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _connected_components(
    adjacency: dict[int, set[int]], excluded: set[int] | None = None
) -> list[list[int]]:
    excluded = excluded or set()
    seen = set(excluded)
    components: list[list[int]] = []

    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in seen or neighbor in excluded:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _label_row_indices(structure_df: pd.DataFrame, row_indices: list[int], role: str) -> None:
    if not row_indices:
        return
    index_labels = structure_df.index.take(row_indices)
    structure_df.loc[index_labels, "atom_role"] = role


def _classify_catalyst_oxygen(structure_df: pd.DataFrame, oxygen_row: int, adjacency: dict[int, set[int]]) -> str:
    carbon_neighbors = [
        neighbor for neighbor in adjacency[oxygen_row] if structure_df.loc[neighbor, "element"] == "C"
    ]
    if not carbon_neighbors:
        return "alkoxide_O"

    carbon_row = carbon_neighbors[0]
    carbon_neighbor_elements = [
        str(structure_df.loc[neighbor, "element"]) for neighbor in adjacency[carbon_row] if neighbor != oxygen_row
    ]
    carbon_neighbor_carbon_count = carbon_neighbor_elements.count("C")
    if carbon_neighbor_carbon_count >= 2:
        return "phenoxy_O"
    return "alkoxide_O"


def _identify_terminal_alkoxide_fragments(
    structure_df: pd.DataFrame, adjacency: dict[int, set[int]]
) -> tuple[set[int], set[int]]:
    alkoxide_oxygen_rows: set[int] = set()
    excluded_rows: set[int] = set()
    metal_element = _clean_optional_token(structure_df["metal_guess"].iloc[0])

    for oxygen_row in structure_df.index[structure_df["element"].eq("O")]:
        carbon_neighbors = [
            neighbor
            for neighbor in adjacency[oxygen_row]
            if structure_df.loc[neighbor, "element"] == "C"
        ]
        for carbon_row in carbon_neighbors:
            carbon_neighbor_elements = [
                str(structure_df.loc[neighbor, "element"])
                for neighbor in adjacency[carbon_row]
                if neighbor != oxygen_row
            ]
            carbon_neighbor_carbon_count = carbon_neighbor_elements.count("C")
            non_h_non_metal_neighbors = [
                element
                for element in carbon_neighbor_elements
                if element not in {"H", metal_element}
            ]
            if carbon_neighbor_carbon_count != 0:
                continue
            if non_h_non_metal_neighbors:
                continue

            alkoxide_oxygen_rows.add(int(oxygen_row))
            excluded_rows.add(int(oxygen_row))
            excluded_rows.add(int(carbon_row))
            break

    return alkoxide_oxygen_rows, excluded_rows


def _element_counts(structure_df: pd.DataFrame, rows: set[int]) -> dict[str, int]:
    if not rows:
        return {}
    return structure_df.loc[sorted(rows), "element"].value_counts().to_dict()


def _clean_optional_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _find_component_with_monomer(
    structure_df: pd.DataFrame,
    adjacency: dict[int, set[int]],
    metal_row: int,
    excluded_rows: set[int] | None = None,
) -> set[int]:
    excluded_rows = excluded_rows or set()
    components = _connected_components(adjacency, excluded={metal_row, *excluded_rows})
    monomer_guess = _clean_optional_token(structure_df["monomer_guess"].iloc[0])
    expected_counts = _EXPECTED_MONOMER_COMPOSITION.get(monomer_guess, {})

    candidate_components: list[set[int]] = []
    for component in components:
        component_set = set(component)
        component_elements = _element_counts(structure_df, component_set)
        if component_elements.get("Fe", 0) > 0:
            continue
        if component_elements.get("C", 0) == 0 or component_elements.get("O", 0) == 0:
            continue
        candidate_components.append(component_set)

    if not candidate_components:
        return set()

    if not expected_counts:
        return max(candidate_components, key=len)

    best_component: set[int] = set()
    best_score: tuple[int, int, int] | None = None
    for r in range(1, len(candidate_components) + 1):
        for subset in itertools.combinations(candidate_components, r):
            combined_rows = set().union(*subset)
            counts = _element_counts(structure_df, combined_rows)
            c_diff = abs(counts.get("C", 0) - expected_counts.get("C", 0))
            o_diff = abs(counts.get("O", 0) - expected_counts.get("O", 0))
            score = (c_diff + o_diff, abs(len(combined_rows) - sum(expected_counts.values())), -len(combined_rows))
            if best_score is None or score < best_score:
                best_score = score
                best_component = combined_rows

    return best_component


def _annotate_structure(structure_df: pd.DataFrame) -> pd.DataFrame:
    structure_df = structure_df.copy().reset_index(drop=True)
    structure_df["atom_role"] = ""

    metal_matches = structure_df.index[structure_df["element"].eq(structure_df["metal_guess"])]
    if len(metal_matches) != 1:
        file_name = str(structure_df["file_name"].iloc[0])
        raise ValueError(f"Expected exactly one metal atom for {file_name}, found {len(metal_matches)}")

    metal_row = int(metal_matches[0])
    structure_df.loc[metal_row, "atom_role"] = "metal_center"

    adjacency = _build_adjacency(structure_df)
    alkoxide_oxygen_rows, alkoxide_excluded_rows = _identify_terminal_alkoxide_fragments(
        structure_df, adjacency
    )

    fe_candidates = sorted(
        (
            row
            for row in structure_df.index
            if structure_df.loc[row, "element"] == "Fe"
        ),
        key=lambda row: (structure_df.loc[row, "atom_index"]),
    )
    if fe_candidates:
        fe_row = fe_candidates[0]
        structure_df.loc[fe_row, "atom_role"] = "ferrocene_Fe"
        ferrocene_carbons = sorted(
            [
                neighbor
                for neighbor in adjacency[fe_row]
                if structure_df.loc[neighbor, "element"] == "C"
            ],
            key=lambda row: structure_df.loc[row, "atom_index"],
        )
        _label_row_indices(structure_df, ferrocene_carbons, "ferrocene_C")

    monomer_guess = _clean_optional_token(structure_df["monomer_guess"].iloc[0])
    monomer_component: set[int] = set()
    if monomer_guess:
        monomer_component = _find_component_with_monomer(
            structure_df, adjacency, metal_row, excluded_rows=alkoxide_excluded_rows
        )
        for row in sorted(monomer_component, key=lambda value: structure_df.loc[value, "atom_index"]):
            element = str(structure_df.loc[row, "element"])
            if element in {"C", "N", "O", "P", "S"}:
                structure_df.loc[row, "atom_role"] = f"monomer_{element}"

    for element, role in DONOR_ROLE_BY_ELEMENT.items():
        donor_rows = sorted(
            [
                row
                for row in structure_df.index
                if structure_df.loc[row, "element"] == element and row not in monomer_component
            ],
            key=lambda row: structure_df.loc[row, "atom_index"],
        )
        _label_row_indices(structure_df, donor_rows, role)

    for row in sorted(structure_df.index, key=lambda value: structure_df.loc[value, "atom_index"]):
        if structure_df.loc[row, "element"] != "O":
            continue
        if structure_df.loc[row, "atom_role"]:
            continue
        if row in alkoxide_oxygen_rows:
            structure_df.loc[row, "atom_role"] = "alkoxide_O"
        else:
            structure_df.loc[row, "atom_role"] = _classify_catalyst_oxygen(
                structure_df, row, adjacency
            )

    return structure_df[KEY_COLUMNS + ["atom_role", "notes"]]


def auto_annotate_by_element(
    atom_table_csv: str, annotation_csv: str, output_csv: str
) -> pd.DataFrame:
    atom_table = pd.read_csv(atom_table_csv)
    annotation = pd.read_csv(annotation_csv)
    annotation["notes"] = annotation["notes"].fillna("").astype(str)
    annotation = _clear_auto_generated_roles(annotation)

    merged = atom_table.merge(
        annotation[KEY_COLUMNS + ["atom_role", "notes"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged["notes"] = merged["notes"].fillna("").astype(str)

    annotated_structures: list[pd.DataFrame] = []
    for _, structure_df in merged.groupby(
        ["dataset_group", "file_name", "structure_name"], sort=False
    ):
        annotated_structures.append(_annotate_structure(structure_df))

    annotation = pd.concat(annotated_structures, ignore_index=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(output_path, index=False)
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-annotate catalyst atoms from inferred molecular connectivity."
    )
    parser.add_argument("atom_table_csv", help="Atom coordinate table CSV")
    parser.add_argument("annotation_csv", help="Existing annotation template CSV")
    parser.add_argument("output_csv", help="Output CSV with auto-filled roles")
    args = parser.parse_args()

    annotated = auto_annotate_by_element(
        args.atom_table_csv, args.annotation_csv, args.output_csv
    )
    filled_count = int((annotated["atom_role"].fillna("").astype(str).str.strip() != "").sum())
    print(f"Wrote {filled_count} annotated rows to {args.output_csv}")


if __name__ == "__main__":
    main()
