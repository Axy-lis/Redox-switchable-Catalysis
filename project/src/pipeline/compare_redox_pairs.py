from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_pair_key(structure_name: str) -> str:
    pair_key = re.sub(r"(?<=-)(ox|red)(?=-|$)", "", structure_name)
    pair_key = re.sub(r"--+", "-", pair_key)
    return pair_key.strip("-")


def _ordered_role_block(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["atom_role", "atom_index"], kind="stable").reset_index(
        drop=True
    )


def _kabsch_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_center = reference.mean(axis=0)
    tgt_center = target.mean(axis=0)
    ref_centered = reference - ref_center
    tgt_centered = target - tgt_center

    covariance = tgt_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    aligned = tgt_centered @ rotation + ref_center
    return aligned


def _rmsd(reference: np.ndarray, target: np.ndarray) -> float:
    diffs = reference - target
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def _role_rmsd(
    reference_df: pd.DataFrame, aligned_target: np.ndarray, role: str
) -> float | None:
    role_mask = reference_df["atom_role"].eq(role).to_numpy()
    if not role_mask.any():
        return None
    role_ref = reference_df.loc[role_mask, ["x", "y", "z"]].to_numpy(dtype=float)
    role_tgt = aligned_target[role_mask]
    return _rmsd(role_ref, role_tgt)


def _role_prefix_rmsd(
    reference_df: pd.DataFrame, aligned_target: np.ndarray, role_prefix: str
) -> float | None:
    role_mask = reference_df["atom_role"].str.startswith(role_prefix).to_numpy()
    if not role_mask.any():
        return None
    role_ref = reference_df.loc[role_mask, ["x", "y", "z"]].to_numpy(dtype=float)
    role_tgt = aligned_target[role_mask]
    return _rmsd(role_ref, role_tgt)


def _role_suffix_rmsd(
    reference_df: pd.DataFrame, aligned_target: np.ndarray, role_suffix: str
) -> float | None:
    role_mask = reference_df["atom_role"].str.endswith(role_suffix).to_numpy()
    if not role_mask.any():
        return None
    role_ref = reference_df.loc[role_mask, ["x", "y", "z"]].to_numpy(dtype=float)
    role_tgt = aligned_target[role_mask]
    return _rmsd(role_ref, role_tgt)


def _mean_distance_to_role(df: pd.DataFrame, source_role: str, target_role: str) -> float | None:
    source = df[df["atom_role"] == source_role][["x", "y", "z"]].to_numpy(dtype=float)
    target = df[df["atom_role"] == target_role][["x", "y", "z"]].to_numpy(dtype=float)
    if len(source) == 0 or len(target) == 0:
        return None
    if len(source) != 1:
        raise ValueError(f"Expected exactly one {source_role}, found {len(source)}")
    distances = np.linalg.norm(target - source[0], axis=1)
    return float(distances.mean())


def _fe_distance(df: pd.DataFrame) -> float | None:
    return _mean_distance_to_role(df, "metal_center", "ferrocene_Fe")


def _mean_distance_to_role_prefix(
    df: pd.DataFrame, source_role: str, target_prefix: str
) -> float | None:
    source = df[df["atom_role"] == source_role][["x", "y", "z"]].to_numpy(dtype=float)
    target = df[df["atom_role"].str.startswith(target_prefix)][["x", "y", "z"]].to_numpy(
        dtype=float
    )
    if len(source) == 0 or len(target) == 0:
        return None
    if len(source) != 1:
        raise ValueError(f"Expected exactly one {source_role}, found {len(source)}")
    distances = np.linalg.norm(target - source[0], axis=1)
    return float(distances.mean())


def _mean_distance_to_role_suffix(
    df: pd.DataFrame, source_role: str, target_suffix: str
) -> float | None:
    source = df[df["atom_role"] == source_role][["x", "y", "z"]].to_numpy(dtype=float)
    target = df[df["atom_role"].str.endswith(target_suffix)][["x", "y", "z"]].to_numpy(
        dtype=float
    )
    if len(source) == 0 or len(target) == 0:
        return None
    if len(source) != 1:
        raise ValueError(f"Expected exactly one {source_role}, found {len(source)}")
    distances = np.linalg.norm(target - source[0], axis=1)
    return float(distances.mean())


def _safe_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b - a)


def compare_redox_pairs(selected_atoms_csv: str, output_csv: str) -> pd.DataFrame:
    selected = pd.read_csv(selected_atoms_csv)
    selected["atom_role"] = selected["atom_role"].fillna("").astype(str)
    selected = selected[selected["atom_role"] != ""].copy()
    if selected.empty:
        raise ValueError(f"No selected atoms found in: {selected_atoms_csv}")

    selected["pair_key"] = selected["structure_name"].map(_build_pair_key)

    rows: list[dict[str, Any]] = []
    for pair_key, pair_df in selected.groupby("pair_key", sort=True):
        ox_df = pair_df[pair_df["redox_state_guess"] == "ox"].copy()
        red_df = pair_df[pair_df["redox_state_guess"] == "red"].copy()
        if ox_df.empty or red_df.empty:
            continue

        ox_ordered = _ordered_role_block(ox_df)
        red_ordered = _ordered_role_block(red_df)

        role_counts_ox = ox_ordered["atom_role"].value_counts().sort_index()
        role_counts_red = red_ordered["atom_role"].value_counts().sort_index()
        if not role_counts_ox.equals(role_counts_red):
            raise ValueError(
                f"Role count mismatch for pair {pair_key}: "
                f"ox={role_counts_ox.to_dict()} red={role_counts_red.to_dict()}"
            )

        role_signature_same = ox_ordered["atom_role"].tolist() == red_ordered["atom_role"].tolist()
        if not role_signature_same:
            raise ValueError(f"Role ordering mismatch for pair {pair_key}")

        ox_coords = ox_ordered[["x", "y", "z"]].to_numpy(dtype=float)
        red_coords = red_ordered[["x", "y", "z"]].to_numpy(dtype=float)
        red_aligned = _kabsch_align(ox_coords, red_coords)

        selected_atom_rmsd = _rmsd(ox_coords, red_aligned)
        metal_center_rmsd = _role_rmsd(ox_ordered, red_aligned, "metal_center")
        donor_n_rmsd = _role_rmsd(ox_ordered, red_aligned, "donor_N")
        donor_atom_rmsd = _role_prefix_rmsd(ox_ordered, red_aligned, "donor_")
        fe_rmsd = _role_rmsd(ox_ordered, red_aligned, "ferrocene_Fe")
        ferrocene_c_rmsd = _role_rmsd(ox_ordered, red_aligned, "ferrocene_C")
        oxygen_atom_rmsd = _role_suffix_rmsd(ox_ordered, red_aligned, "_O")
        phenoxy_o_rmsd = _role_rmsd(ox_ordered, red_aligned, "phenoxy_O")
        alkoxide_o_rmsd = _role_rmsd(ox_ordered, red_aligned, "alkoxide_O")
        monomer_o_rmsd = _role_rmsd(ox_ordered, red_aligned, "monomer_O")
        monomer_c_rmsd = _role_rmsd(ox_ordered, red_aligned, "monomer_C")

        ox_m_n = _mean_distance_to_role(ox_ordered, "metal_center", "donor_N")
        red_m_n = _mean_distance_to_role(red_ordered, "metal_center", "donor_N")
        ox_m_donor = _mean_distance_to_role_prefix(ox_ordered, "metal_center", "donor_")
        red_m_donor = _mean_distance_to_role_prefix(red_ordered, "metal_center", "donor_")
        ox_m_o = _mean_distance_to_role_suffix(ox_ordered, "metal_center", "_O")
        red_m_o = _mean_distance_to_role_suffix(red_ordered, "metal_center", "_O")
        ox_m_phenoxy = _mean_distance_to_role(ox_ordered, "metal_center", "phenoxy_O")
        red_m_phenoxy = _mean_distance_to_role(red_ordered, "metal_center", "phenoxy_O")
        ox_m_alkoxide = _mean_distance_to_role(ox_ordered, "metal_center", "alkoxide_O")
        red_m_alkoxide = _mean_distance_to_role(red_ordered, "metal_center", "alkoxide_O")
        ox_m_monomer_o = _mean_distance_to_role(ox_ordered, "metal_center", "monomer_O")
        red_m_monomer_o = _mean_distance_to_role(red_ordered, "metal_center", "monomer_O")
        ox_m_fe = _fe_distance(ox_ordered)
        red_m_fe = _fe_distance(red_ordered)

        rows.append(
            {
                "pair_key": pair_key,
                "metal_guess": ox_ordered["metal_guess"].iloc[0],
                "ox_file": ox_ordered["file_name"].iloc[0],
                "red_file": red_ordered["file_name"].iloc[0],
                "selected_atom_count": len(ox_ordered),
                "selected_atom_rmsd": selected_atom_rmsd,
                "metal_center_rmsd": metal_center_rmsd,
                "donor_atom_rmsd": donor_atom_rmsd,
                "donor_N_rmsd": donor_n_rmsd,
                "oxygen_atom_rmsd": oxygen_atom_rmsd,
                "phenoxy_O_rmsd": phenoxy_o_rmsd,
                "alkoxide_O_rmsd": alkoxide_o_rmsd,
                "monomer_O_rmsd": monomer_o_rmsd,
                "ferrocene_Fe_rmsd": fe_rmsd,
                "ferrocene_C_rmsd": ferrocene_c_rmsd,
                "monomer_C_rmsd": monomer_c_rmsd,
                "mean_metal_to_any_donor_ox": ox_m_donor,
                "mean_metal_to_any_donor_red": red_m_donor,
                "mean_metal_to_any_donor_change": _safe_delta(ox_m_donor, red_m_donor),
                "mean_metal_to_donor_N_ox": ox_m_n,
                "mean_metal_to_donor_N_red": red_m_n,
                "mean_metal_to_donor_N_change": _safe_delta(ox_m_n, red_m_n),
                "mean_metal_to_any_oxygen_ox": ox_m_o,
                "mean_metal_to_any_oxygen_red": red_m_o,
                "mean_metal_to_any_oxygen_change": _safe_delta(ox_m_o, red_m_o),
                "mean_metal_to_phenoxy_O_ox": ox_m_phenoxy,
                "mean_metal_to_phenoxy_O_red": red_m_phenoxy,
                "mean_metal_to_phenoxy_O_change": _safe_delta(ox_m_phenoxy, red_m_phenoxy),
                "mean_metal_to_alkoxide_O_ox": ox_m_alkoxide,
                "mean_metal_to_alkoxide_O_red": red_m_alkoxide,
                "mean_metal_to_alkoxide_O_change": _safe_delta(ox_m_alkoxide, red_m_alkoxide),
                "mean_metal_to_monomer_O_ox": ox_m_monomer_o,
                "mean_metal_to_monomer_O_red": red_m_monomer_o,
                "mean_metal_to_monomer_O_change": _safe_delta(ox_m_monomer_o, red_m_monomer_o),
                "metal_to_Fe_ox": ox_m_fe,
                "metal_to_Fe_red": red_m_fe,
                "metal_to_Fe_change": _safe_delta(ox_m_fe, red_m_fe),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(by=["metal_guess", "pair_key"]).reset_index(
        drop=True
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare oxidized and reduced catalyst pairs using selected atoms."
    )
    parser.add_argument("selected_atoms_csv", help="CSV containing selected atoms")
    parser.add_argument("output_csv", help="Output CSV for ox/red comparison metrics")
    args = parser.parse_args()

    comparison = compare_redox_pairs(args.selected_atoms_csv, args.output_csv)
    print(f"Wrote {len(comparison)} redox-pair comparison rows to {args.output_csv}")


if __name__ == "__main__":
    main()
