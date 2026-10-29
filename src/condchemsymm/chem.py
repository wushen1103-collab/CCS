from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol


def canonical_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def canonical_unmapped_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def connectivity_key(smiles: str) -> Optional[str]:
    return canonical_smiles(smiles, isomeric=False)


def scaffold_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "PARSE_FAIL"
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaf = ""
    return scaf or connectivity_key(smiles) or "EMPTY_SCAFFOLD"


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2, include_chirality: bool = True) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        useChirality=include_chirality,
    )
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_hash_fp(smiles: str, n_bits: int = 2048, min_n: int = 2, max_n: int = 4) -> np.ndarray:
    """Simple signed character n-gram hash vector.

    This intentionally keeps representation sensitivity to SMILES spelling so
    Phase 0 can test whether positive-view aggregation helps.
    """
    text = str(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % n_bits
            sign = 1.0 if (val >> 63) == 0 else -1.0
            arr[idx] += sign
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


def random_smiles(smiles: str, seed: int = 0) -> Optional[str]:
    """Deterministic random-like SMILES via seeded atom renumbering."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    order = list(range(mol.GetNumAtoms()))
    rng = random.Random(seed + 1009)
    rng.shuffle(order)
    renum = Chem.RenumberAtoms(mol, order)
    return Chem.MolToSmiles(renum, canonical=False, isomericSmiles=True)


def atom_renumber_smiles(smiles: str, seed: int = 0) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    order = list(range(mol.GetNumAtoms()))
    rng = random.Random(seed)
    rng.shuffle(order)
    renum = Chem.RenumberAtoms(mol, order)
    return Chem.MolToSmiles(renum, canonical=False, isomericSmiles=True)


def invert_chirality_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    changed = False
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
            changed = True
        elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            changed = True
    if not changed:
        return None
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if out == canonical_smiles(smiles, isomeric=True):
        return None
    return out


def chiral_centers(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    except TypeError:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    return [(idx, cip) for idx, cip in centers if cip in {"R", "S"}]


def single_rs_label(smiles: str) -> Optional[int]:
    centers = chiral_centers(smiles)
    if len(centers) != 1:
        return None
    return 1 if centers[0][1] == "R" else 0


@dataclass(frozen=True)
class DatasetColumns:
    smiles: str
    label: str
