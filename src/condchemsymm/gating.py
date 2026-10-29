from __future__ import annotations


REPRESENTATION_TRANSFORMS = {"identity", "atom_renumber", "random_smiles"}


def mechanistic_accepts(dataset: str, task_type: str, transform_type: str) -> bool:
    """Task/chemistry semantic gate used as a non-oracle Phase 0 prototype."""
    if transform_type in REPRESENTATION_TRANSFORMS:
        return True
    if transform_type == "tautomer_shift":
        return False
    if transform_type == "mirror_chirality":
        return dataset != "synthetic_rs"
    return False


def mechanistic_accepts_edge(edge_row, task_type: str = "") -> bool:
    return mechanistic_accepts(
        dataset=str(edge_row.get("dataset", "")),
        task_type=str(edge_row.get("task_type", task_type)),
        transform_type=str(edge_row.get("transform_type", "")),
    )
