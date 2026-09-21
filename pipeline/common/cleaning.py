"""
Cleaning rules and polymer keys. The same rules are applied to the real data
(prepare_real_data.py) and to the two synthetic sets (clean_synthetic_data.py).

Two polymers count as the same polymer when they match on EITHER of two keys:
- the exact RDKit canonical SMILES of the repeat unit, or
- the ring-closure key, which is the same for every way of cutting one chain into a repeat unit.
This "either key" rule is used everywhere overlap or duplication is checked: deduplication of
the real and the synthetic data, removal of test and validation polymers, the memorization
counts, and the deduplication of mixed training sets.
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
    repeat unit such as *C*), when the two neighbours are already bonded (*CC*), or when the
    SMILES has more than one fragment. It does not recognize a repeat unit written twice
    (*CCCC* against *CC*).
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or "." in smiles:
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


def polymer_keys(smiles):
    """Both keys of a polymer: (exact canonical SMILES, ring-closure key or None)."""
    exact = canonical_smiles(smiles)
    return exact, (ring_closure_key(exact) if exact is not None else None)


def polymer_id(smiles):
    """
    One label per polymer for grouping duplicates: the ring key when it exists, otherwise the
    exact canonical SMILES. Because the ring key is computed from the canonical SMILES, two
    rows share this label exactly when they match on either key, so grouping by it is the
    "either key" rule. (A ring key has no * and can never equal a canonical repeat unit.)
    """
    exact, ring = polymer_keys(smiles)
    return ring if ring is not None else exact


class KeySet:
    """The keys of a set of polymers, to ask whether another polymer matches any of them on either key."""

    def __init__(self, smiles_list):
        keys = [polymer_keys(s) for s in smiles_list]
        self.exact = {e for e, _ in keys if e is not None}
        self.ring = {r for _, r in keys if r is not None}

    def match_type(self, smiles):
        """ "exact", "ring" (same polymer, different cut) or None."""
        exact, ring = polymer_keys(smiles)
        if exact in self.exact:
            return "exact"
        if ring is not None and ring in self.ring:
            return "ring"
        return None

    def matches(self, smiles_series):
        """Boolean Series: True where the polymer matches the set on either key."""
        return smiles_series.map(lambda s: self.match_type(s) is not None)


def merge_duplicates(df):
    """
    Merge rows that are the same polymer on either key: median Tg, and the alphabetically
    first canonical SMILES of the group as its one spelling. Returns columns smiles, tg_celsius.
    """
    df = df.assign(polymer_id=df["smiles"].map(polymer_id)).sort_values("smiles")
    merged = df.groupby("polymer_id", as_index=False).agg(smiles=("smiles", "first"), tg_celsius=("tg_celsius", "median"))
    return merged[["smiles", "tg_celsius"]].sort_values("smiles").reset_index(drop=True)


def clean_dataset(df, held_out):
    """
    Cleaning of a synthetic set. df has columns "smiles" and "tg_celsius"; held_out is a
    dict {"test": KeySet, "validation": KeySet}. Rules 1-4 are row filters; rule 5 merges
    duplicates on either key and rule 6 removes every polymer that matches the test or the
    validation set on either key. The count of unique exact SMILES is kept in the table so
    the effect of the ring key is visible.
    Returns (clean DataFrame, dict of row counts after each step).
    """
    counts = {"0_raw": len(df)}
    df = df.copy()
    df["tg_celsius"] = pd.to_numeric(df["tg_celsius"], errors="coerce")
    df = df.dropna(subset=["tg_celsius"])
    counts["1_numeric_tg"] = len(df)

    df["smiles"] = df["smiles"].map(canonical_smiles)
    df = df.dropna(subset=["smiles"])
    counts["2_valid_smiles"] = len(df)

    df = df[df["smiles"].str.count(r"\*") == 2]
    if config.REQUIRE_SINGLE_FRAGMENT:
        df = df[~df["smiles"].str.contains(".", regex=False)]
    counts["3_two_stars"] = len(df)

    in_range = (df["tg_celsius"] >= config.TG_MIN_CELSIUS) & (df["tg_celsius"] <= config.TG_MAX_CELSIUS)
    df = df[in_range]
    counts["4_tg_in_range"] = len(df)

    counts["5a_unique_exact_smiles"] = df["smiles"].nunique()       # exact matching alone
    df = merge_duplicates(df)
    counts["5b_unique_either_key"] = len(df)

    for name in ["test", "validation"]:
        match = df["smiles"].map(held_out[name].match_type)
        counts[f"removed_{name}_exact"] = int((match == "exact").sum())
        counts[f"removed_{name}_ring"] = int((match == "ring").sum())
        df = df[match.isna()]
        counts["6a_not_in_test" if name == "test" else "6b_not_in_validation"] = len(df)

    return df.reset_index(drop=True), counts


def print_counts(counts, name):
    """Print the row count after each step and how many rows the step removed."""
    print(f"\nCleaning {name}:")
    previous = None
    for step, n in counts.items():
        if step.startswith("removed_"):     # detail lines of rule 6, not a row count
            print(f"      {step:<26} {n:>7}")
            continue
        removed = "" if previous is None else f"  (removed {previous - n})"
        print(f"  {step:<26} {n:>7}{removed}")
        previous = n
