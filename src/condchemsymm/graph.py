from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem

from condchemsymm.chem import mol_from_smiles


ATOM_NUMS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot_unknown(value, choices):
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot_unknown(atom.GetAtomicNum(), ATOM_NUMS)
        + one_hot_unknown(atom.GetTotalDegree(), DEGREES)
        + one_hot_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot_unknown(atom.GetChiralTag(), CHIRAL_TAGS)
        + one_hot_unknown(atom.GetTotalNumHs(), NUM_HS)
        + one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.HasProp("_CIPCode")),
            float(atom.GetMass() * 0.01),
        ]
    )


def bond_features(bond: Chem.Bond | None) -> list[float]:
    if bond is None:
        return [1.0] + [0.0] * (len(BOND_TYPES) + 1 + 2 + len(BOND_STEREO) + 1)
    return (
        [0.0]
        + one_hot_unknown(bond.GetBondType(), BOND_TYPES)
        + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        + one_hot_unknown(bond.GetStereo(), BOND_STEREO)
    )


ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FDIM = len(bond_features(None))


@dataclass
class MolGraph:
    atom_x: np.ndarray
    bond_x: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    rev: np.ndarray


def mol_graph_from_smiles(smiles: str) -> MolGraph:
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        mol = Chem.MolFromSmiles("C")
    atoms = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    bond_x = []
    src = []
    dst = []
    rev = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        f = bond_features(bond)
        e0 = len(src)
        src.append(a)
        dst.append(b)
        bond_x.append(f)
        e1 = len(src)
        src.append(b)
        dst.append(a)
        bond_x.append(f)
        rev.extend([e1, e0])

    if not src:
        # Single-atom molecule: add no directed bonds and let readout use atom features.
        return MolGraph(
            atom_x=atoms,
            bond_x=np.zeros((0, BOND_FDIM), dtype=np.float32),
            src=np.zeros((0,), dtype=np.int64),
            dst=np.zeros((0,), dtype=np.int64),
            rev=np.zeros((0,), dtype=np.int64),
        )

    return MolGraph(
        atom_x=atoms,
        bond_x=np.asarray(bond_x, dtype=np.float32),
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        rev=np.asarray(rev, dtype=np.int64),
    )


def collate_mol_graphs(graphs: list[MolGraph], device: torch.device):
    atom_parts = []
    bond_parts = []
    src_parts = []
    dst_parts = []
    rev_parts = []
    atom_scope = []
    atom_offset = 0
    bond_offset = 0
    for g in graphs:
        atom_parts.append(torch.as_tensor(g.atom_x, dtype=torch.float32))
        bond_parts.append(torch.as_tensor(g.bond_x, dtype=torch.float32))
        if len(g.src):
            src_parts.append(torch.as_tensor(g.src + atom_offset, dtype=torch.long))
            dst_parts.append(torch.as_tensor(g.dst + atom_offset, dtype=torch.long))
            rev_parts.append(torch.as_tensor(g.rev + bond_offset, dtype=torch.long))
        atom_scope.append((atom_offset, len(g.atom_x)))
        atom_offset += len(g.atom_x)
        bond_offset += len(g.bond_x)

    atom_x = torch.cat(atom_parts, dim=0).to(device)
    bond_x = torch.cat(bond_parts, dim=0).to(device) if bond_parts else torch.zeros((0, BOND_FDIM), device=device)
    src = torch.cat(src_parts, dim=0).to(device) if src_parts else torch.zeros((0,), dtype=torch.long, device=device)
    dst = torch.cat(dst_parts, dim=0).to(device) if dst_parts else torch.zeros((0,), dtype=torch.long, device=device)
    rev = torch.cat(rev_parts, dim=0).to(device) if rev_parts else torch.zeros((0,), dtype=torch.long, device=device)
    return {
        "atom_x": atom_x,
        "bond_x": bond_x,
        "src": src,
        "dst": dst,
        "rev": rev,
        "atom_scope": atom_scope,
    }
