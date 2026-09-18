"""
Cleaning rules for the synthetic datasets. Used by clean_synthetic_data.py, and by
generate_with_claude.py to count clean pairs between generation rounds, so both
always apply exactly the same rules.
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

import config

# RDKit prints a warning for every SMILES it cannot parse; we count them instead.
RDLogger.DisableLog("rdApp.*")


def canonical_smiles(smiles):
    """Return the RDKit canonical SMILES, or None if RDKit cannot parse it. The * atoms are kept."""
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def ring_closure_key(smiles):
    """
    A key that is the same for every way of cutting the same polymer chain into a repeat unit.
    Joins the two * neighbours with a bond, deletes the * atoms and canonicalizes, so a repeat
    unit written from a different backbone atom gives the same ring. Returns None when the
    SMILES has no two degree-one * atoms, when both stars sit on the same atom (a one-atom
    repeat unit such as *C*), or when the two neighbours are already bonded. Not used by
    clean_dataset (the published run used exact canonical matching); used by the paper
    scripts to count boundary-shifted duplicates that exact matching misses.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2 or any(mol.GetAtomWithIdx(i).GetDegree() != 1 for i in stars):
        return None
    neighbours = [mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in stars]
    if neighbours[0] == neighbours[1] or mol.GetBondBetweenAtoms(*neighbours) is not None:
        return None
    editable = Chem.RWMol(mol)
    editable.AddBond(neighbours[0], neighbours[1], mol.GetBondBetweenAtoms(stars[0], neighbours[0]).GetBondType())
    for i in sorted(stars, reverse=True):
        editable.RemoveAtom(i)
    try:
        Chem.SanitizeMol(editable)
    except Exception:
        return None
    return Chem.MolToSmiles(editable)


def clean_dataset(df, test_smiles):
    """
    Apply the cleaning steps to a DataFrame with columns "smiles" and "tg_celsius".
    test_smiles is a set of canonical SMILES from test_real.csv.
    Returns (clean DataFrame with columns smiles, tg_celsius) and a dict of row counts
    after each step, so the script can print how many rows each step removed.
    """
    counts = {"0_raw": len(df)}

    # Step 1: drop missing or non-numeric Tg (invalid JSON never becomes a row, see parsing).
    df = df.copy()
    df["tg_celsius"] = pd.to_numeric(df["tg_celsius"], errors="coerce")
    df = df.dropna(subset=["tg_celsius"])
    counts["1_numeric_tg"] = len(df)

    # Step 2: drop SMILES that RDKit cannot parse; canonicalize the rest.
    df["smiles"] = df["smiles"].map(canonical_smiles)
    df = df.dropna(subset=["smiles"])
    counts["2_valid_smiles"] = len(df)

    # Step 3: a repeat unit has exactly two connection points. Fragments with one * or
    # branched structures with three * are not polymers and are dropped.
    # This counts * characters; it does not check that both sit in one connected
    # molecule, so a SMILES like "*.*N1C(=O)..." (a lone * plus a one-star fragment) or
    # "*C(=O)CCCN.*OCC..." (two one-star fragments) passes. The published run kept 36 such
    # rows in the generated set; config.REQUIRE_SINGLE_FRAGMENT = True removes them.
    df = df[df["smiles"].str.count(r"\*") == 2]
    if config.REQUIRE_SINGLE_FRAGMENT:
        df = df[~df["smiles"].str.contains(".", regex=False)]
    counts["3_two_stars"] = len(df)

    # Step 4: drop Tg values outside the plausible range.
    in_range = (df["tg_celsius"] >= config.TG_MIN_CELSIUS) & (df["tg_celsius"] <= config.TG_MAX_CELSIUS)
    df = df[in_range]
    counts["4_tg_in_range"] = len(df)

    # Step 5: merge duplicate SMILES by taking the median Tg.
    df = df.groupby("smiles", as_index=False)["tg_celsius"].median()
    counts["5_unique_smiles"] = len(df)

    # Step 6: remove every polymer that appears in the test set.
    df = df[~df["smiles"].isin(test_smiles)].reset_index(drop=True)
    counts["6_not_in_test"] = len(df)

    return df, counts


def print_counts(counts, name):
    """Print the row count after each step and how many rows the step removed."""
    print(f"\nCleaning {name}:")
    previous = None
    for step, n in counts.items():
        removed = "" if previous is None else f"  (removed {previous - n})"
        print(f"  {step:<18} {n:>7}{removed}")
        previous = n
