from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi


RDLogger.DisableLog("rdApp.*")


def canonical_smiles(value: object) -> str | None:
    mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def connectivity_key(value: object) -> str | None:
    mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def neutral_single_fragment(value: object) -> bool:
    mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    return bool(mol and len(Chem.GetMolFrags(mol)) == 1 and Chem.GetFormalCharge(mol) == 0)


def source_states(frame: pd.DataFrame, columns: list[str]) -> tuple[set[str], set[str]]:
    values = pd.concat([frame[column] for column in columns], ignore_index=True).dropna()
    exact = set(values.map(canonical_smiles).dropna())
    families = set(values.map(connectivity_key).dropna())
    return exact, families


def source_pairs(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, str]]:
    if len(columns) != 2:
        return set()
    pairs: set[tuple[str, str]] = set()
    for left, right in zip(frame[columns[0]], frame[columns[1]]):
        a, b = canonical_smiles(left), canonical_smiles(right)
        if a and b:
            pairs.add(tuple(sorted((a, b))))
    return pairs


def pair_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_key, group in samples.groupby("family_key", sort=True):
        if len(group) != 2:
            continue
        group = group.sort_values("sample_id")
        left, right = group.iloc[0], group.iloc[1]
        rows.append(
            {
                "family_key": family_key,
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_canonical": canonical_smiles(left["smiles"]),
                "right_canonical": canonical_smiles(right["smiles"]),
                "left_connectivity": connectivity_key(left["smiles"]),
                "right_connectivity": connectivity_key(right["smiles"]),
            }
        )
    out = pd.DataFrame(rows)
    out["pair_key"] = out.apply(
        lambda row: "||".join(sorted((str(row["left_canonical"]), str(row["right_canonical"])))), axis=1
    )
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6g") if not frame.empty else "_No rows._"


def build_edges(samples: pd.DataFrame, dataset: str) -> pd.DataFrame:
    by_id = samples.set_index("sample_id")
    rows = []
    for row in samples.itertuples(index=False):
        rows.append(
            {
                "sample_id": row.sample_id,
                "dataset": dataset,
                "base_smiles": row.smiles,
                "target_y": float(row.y),
                "view_id": f"{row.sample_id}:identity",
                "transform_type": "identity",
                "view_smiles": row.smiles,
                "z": 1,
                "evidence": "identity",
            }
        )
        if row.paired_id in by_id.index:
            paired = by_id.loc[row.paired_id]
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "dataset": dataset,
                    "base_smiles": row.smiles,
                    "target_y": float(row.y),
                    "view_id": f"{row.sample_id}:tautomer_shift",
                    "transform_type": "tautomer_shift",
                    "view_smiles": paired["smiles"],
                    "z": 0,
                    "evidence": "paired_experimental_tautomer_preference",
                }
            )
    return pd.DataFrame(rows)


def build_sphysnet_100(root: Path) -> pd.DataFrame:
    path = root / "data/raw/external/sPhysNet-Taut-src/exp_data/results_of_100_tautomers_set.csv"
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        ddg = float(row["ddG"])
        if not np.isfinite(ddg) or np.isclose(ddg, 0.0):
            continue
        pair_id = str(row["name"])
        left_id, right_id = f"sphys100_{pair_id}_t1", f"sphys100_{pair_id}_t2"
        left_y = 1.0 if ddg > 0.0 else 0.0
        right_y = 1.0 - left_y
        common = {
            "dataset": "sphysnet_100_aqueous",
            "task_type": "classification",
            "family_key": f"sphys100_{pair_id}",
            "split": "external",
            "source_name": pair_id,
            "source_row": int(row["Row Nr"]),
            "log_k": float(row["log K"]),
            "experimental_ddg_kcal_mol": ddg,
            "sphysnet_published_ddg": float(row["sphysnet-pre"]),
            "solvent_scope": "aqueous_solution",
        }
        rows.append({**common, "sample_id": left_id, "paired_id": right_id, "smiles": row["smiles0"], "y": left_y})
        rows.append({**common, "sample_id": right_id, "paired_id": left_id, "smiles": row["smiles1"], "y": right_y})
    return pd.DataFrame(rows)


def build_nci_candidates(root: Path, excluded_families: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "data/raw/external/NCI-Tautomer-Database/Tautomer_database_release_3a.xlsx"
    raw = pd.read_excel(path, sheet_name="Tautomer_database")
    for column in ["Prevalence_Category_1", "Prevalence_Category_2"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    aqueous = raw["Solvent"].fillna("").astype(str).str.contains(
        r"water|H2O|D2O|aqueous", case=False, regex=True
    )
    pair = raw[raw["Size"].eq(2) & aqueous].copy()
    pair["eligible_explicit_preference"] = (
        pair["Prevalence_Category_1"].notna()
        & pair["Prevalence_Category_2"].notna()
        & pair["Prevalence_Category_1"].ne(pair["Prevalence_Category_2"])
    )
    pair["eligible_neutral_single_fragment"] = pair["SMILES_1"].map(neutral_single_fragment) & pair[
        "SMILES_2"
    ].map(neutral_single_fragment)
    pair["canonical_1"] = pair["SMILES_1"].map(canonical_smiles)
    pair["canonical_2"] = pair["SMILES_2"].map(canonical_smiles)
    pair["family_1"] = pair["SMILES_1"].map(connectivity_key)
    pair["family_2"] = pair["SMILES_2"].map(connectivity_key)
    pair["eligible_distinct_states"] = pair["canonical_1"].notna() & pair["canonical_2"].notna() & pair[
        "canonical_1"
    ].ne(pair["canonical_2"])
    pair["known_source_family_overlap"] = pair["family_1"].isin(excluded_families) | pair["family_2"].isin(
        excluded_families
    )
    pair["pair_key"] = pair.apply(
        lambda row: "||".join(sorted((str(row["canonical_1"]), str(row["canonical_2"])))), axis=1
    )
    pair["duplicate_pair"] = pair.duplicated("pair_key", keep="first")
    pair["included"] = (
        pair["eligible_explicit_preference"]
        & pair["eligible_neutral_single_fragment"]
        & pair["eligible_distinct_states"]
        & ~pair["known_source_family_overlap"]
        & ~pair["duplicate_pair"]
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons = []
        if not row["eligible_explicit_preference"]:
            reasons.append("no_unique_explicit_preference")
        if not row["eligible_neutral_single_fragment"]:
            reasons.append("charged_fragmented_or_invalid")
        if not row["eligible_distinct_states"]:
            reasons.append("invalid_or_identical_states")
        if row["known_source_family_overlap"]:
            reasons.append("known_checkpoint_or_tautobase_family_overlap")
        if row["duplicate_pair"]:
            reasons.append("duplicate_pair_conditions")
        return ";".join(reasons) if reasons else "included"

    pair["exclusion_reason"] = pair.apply(exclusion_reason, axis=1)
    selected = pair[pair["included"]].copy()
    rows = []
    for row in selected.itertuples(index=False):
        ref = int(row.Ref)
        left_id, right_id = f"nci_{ref}_t1", f"nci_{ref}_t2"
        left_y = 1.0 if row.Prevalence_Category_1 > row.Prevalence_Category_2 else 0.0
        right_y = 1.0 - left_y
        common = {
            "dataset": "nci_aqueous_zero_overlap",
            "task_type": "classification",
            "family_key": f"nci_{ref}",
            "split": "external",
            "source_row": ref,
            "solvent": str(row.Solvent),
            "temperature": str(row.Temperature),
            "pH": str(row.pH),
            "experimental_method": str(row.Experimental_Method),
            "publication_doi": str(row.Publication_DOI_1),
            "prevalence_category_1": float(row.Prevalence_Category_1),
            "prevalence_category_2": float(row.Prevalence_Category_2),
            "solvent_scope": "aqueous_or_aqueous_mixture_ood",
        }
        rows.append({**common, "sample_id": left_id, "paired_id": right_id, "smiles": row.SMILES_1, "y": left_y})
        rows.append({**common, "sample_id": right_id, "paired_id": left_id, "smiles": row.SMILES_2, "y": right_y})
    audit_columns = [
        "Ref",
        "Solvent",
        "Temperature",
        "pH",
        "SMILES_1",
        "SMILES_2",
        "Prevalence_Category_1",
        "Prevalence_Category_2",
        "known_source_family_overlap",
        "duplicate_pair",
        "included",
        "exclusion_reason",
    ]
    return pd.DataFrame(rows), pair[audit_columns]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "reports/tables"
    processed_dir = root / "data/processed/evaluation_required"
    raw_nci = root / "data/raw/external/NCI-Tautomer-Database/Tautomer_database_release_3a.xlsx"
    table_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    main_samples = pd.read_parquet(root / "data/processed/main/tautobase_pref_samples.parquet")
    test_samples = main_samples[main_samples["split"].eq("test")].copy()
    test_pairs = pair_table(test_samples)

    source_specs = [
        {
            "checkpoint": "Tautomer-Predictor",
            "role": "validation_model_selection",
            "path": root / "data/raw/external/Tautomer-Predictor-src/data/valid_set_tautobase.csv",
            "columns": ["smiles"],
            "exclude": True,
        },
        {
            "checkpoint": "Tautomer-Predictor",
            "role": "released_published_evaluation_sets",
            "path": root / "data/raw/external/Tautomer-Predictor-src/data/all_test_set.csv",
            "columns": ["smiles"],
            "exclude": True,
        },
        {
            "checkpoint": "sPhysNet-Taut",
            "role": "fine_tune_train",
            "path": root / "data/raw/external/sPhysNet-Taut-src/fine_tune/fold_1/fine_tune_train.csv",
            "columns": ["smiles0", "smiles1"],
            "exclude": True,
        },
        {
            "checkpoint": "sPhysNet-Taut",
            "role": "fine_tune_validation_model_selection",
            "path": root / "data/raw/external/sPhysNet-Taut-src/fine_tune/fold_1/fine_tune_valid.csv",
            "columns": ["smiles0", "smiles1"],
            "exclude": True,
        },
        {
            "checkpoint": "sPhysNet-Taut",
            "role": "published_preference_evaluation",
            "path": root / "data/raw/external/sPhysNet-Taut-src/exp_data/the_preference_test_set.csv",
            "columns": ["smiles0", "smiles1"],
            "exclude": True,
        },
        {
            "checkpoint": "sPhysNet-Taut",
            "role": "published_100_pair_evaluation",
            "path": root / "data/raw/external/sPhysNet-Taut-src/exp_data/results_of_100_tautomers_set.csv",
            "columns": ["smiles0", "smiles1"],
            "exclude": True,
        },
    ]

    overlap_rows = [
        {
            "checkpoint": "Tautomer-Predictor",
            "source_role": "CSD_training_structures",
            "source_path": "not_public_under_CSD_license; released identifiers only",
            "source_rows": np.nan,
            "exact_test_state_rows": np.nan,
            "exact_test_families": np.nan,
            "connectivity_test_families": np.nan,
            "exact_pair_hits": np.nan,
            "audit_status": "not_fully_auditable",
            "clean_subset_action": "qualify checkpoint provenance; cannot certify CSD structure-level non-overlap",
        }
    ]
    exclusion_reasons: dict[str, list[str]] = {key: [] for key in test_pairs["family_key"]}
    known_exact: set[str] = set()
    known_families: set[str] = set()

    for spec in source_specs:
        frame = pd.read_csv(spec["path"])
        exact, families = source_states(frame, spec["columns"])
        pairs = source_pairs(frame, spec["columns"])
        known_exact |= exact
        known_families |= families
        exact_mask = test_samples["smiles"].map(canonical_smiles).isin(exact)
        family_mask = test_samples["smiles"].map(connectivity_key).isin(families)
        pair_keys = {"||".join(pair) for pair in pairs}
        pair_mask = test_pairs["pair_key"].isin(pair_keys)
        overlap_rows.append(
            {
                "checkpoint": spec["checkpoint"],
                "source_role": spec["role"],
                "source_path": str(spec["path"].relative_to(root)),
                "source_rows": len(frame),
                "exact_test_state_rows": int(exact_mask.sum()),
                "exact_test_families": int(test_samples.loc[exact_mask, "family_key"].nunique()),
                "connectivity_test_families": int(test_samples.loc[family_mask, "family_key"].nunique()),
                "exact_pair_hits": int(pair_mask.sum()),
                "audit_status": "audited_local_release_artifact",
                "clean_subset_action": "exclude matching family" if spec["exclude"] else "report_only",
            }
        )
        if spec["exclude"]:
            impacted = set(test_samples.loc[exact_mask | family_mask, "family_key"]) | set(
                test_pairs.loc[pair_mask, "family_key"]
            )
            for family_key in impacted:
                exclusion_reasons[family_key].append(f"{spec['checkpoint']}:{spec['role']}")

    manifest = test_pairs[["family_key", "left_sample_id", "right_sample_id"]].copy()
    manifest["exclusion_reasons"] = manifest["family_key"].map(
        lambda key: ";".join(sorted(set(exclusion_reasons[key])))
    )
    manifest["checkpoint_clean"] = manifest["exclusion_reasons"].eq("")
    clean_ids = set(manifest.loc[manifest["checkpoint_clean"], "left_sample_id"]) | set(
        manifest.loc[manifest["checkpoint_clean"], "right_sample_id"]
    )
    clean_samples = test_samples[test_samples["sample_id"].isin(clean_ids)].copy()
    clean_samples.to_parquet(processed_dir / "tautobase_checkpoint_clean_test_samples.parquet", index=False)
    manifest.to_csv(table_dir / "evaluation_clean_tautobase_family_manifest.csv", index=False)

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(table_dir / "evaluation_checkpoint_overlap_audit.csv", index=False)
    audit_md = [
        "# evaluation Checkpoint Provenance and Overlap Audit",
        "",
        "Canonical-SMILES matching preserves tautomer state. Connectivity matching uses the first block of Standard InChIKey and is used only as a conservative family screen. The clean subset excludes every family matching any locally released train, validation, or published evaluation artifact. Tautomer-Predictor's CSD training structures cannot be fully checked because the structures are not publicly redistributed under the CSD license.",
        "",
        markdown_table(overlap),
        "",
        f"Frozen Tautobase test: {test_samples['family_key'].nunique()} families / {len(test_samples)} states.",
        f"Conservative checkpoint-clean subset: {manifest['checkpoint_clean'].sum()} families / {len(clean_samples)} states.",
        "",
    ]
    (table_dir / "evaluation_checkpoint_overlap_audit.md").write_text("\n".join(audit_md), encoding="utf-8")

    sphys100 = build_sphysnet_100(root)
    sphys100.to_parquet(processed_dir / "sphysnet_100_aqueous_samples.parquet", index=False)
    build_edges(sphys100, "sphysnet_100_aqueous").to_parquet(
        processed_dir / "sphysnet_100_aqueous_edges.parquet", index=False
    )

    main_exact, main_families = source_states(main_samples, ["smiles"])
    nci_excluded_families = known_families | main_families
    nci, nci_audit = build_nci_candidates(root, nci_excluded_families)
    nci.to_parquet(processed_dir / "nci_aqueous_zero_overlap_samples.parquet", index=False)
    build_edges(nci, "nci_aqueous_zero_overlap").to_parquet(
        processed_dir / "nci_aqueous_zero_overlap_edges.parquet", index=False
    )
    nci_audit.to_csv(table_dir / "evaluation_nci_aqueous_selection_audit.csv", index=False)

    sha256 = hashlib.sha256(raw_nci.read_bytes()).hexdigest()
    provenance = {
        "dataset": "NCI/CADD Tautomer Database Release 3a",
        "url": "https://cactus.nci.nih.gov/download/tautomer/Tautomer_database_release_3a.xlsx",
        "downloaded_at": "2026-10-27T13:14:59+08:00",
        "sha256": sha256,
        "bytes": raw_nci.stat().st_size,
        "source_page": "https://cactus.nci.nih.gov/download/tautomer/",
        "license_note": "No explicit redistribution license was located on the download page; retain source attribution and verify terms before redistribution.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = raw_nci.parent / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    external_summary = pd.DataFrame(
        [
            {
                "dataset": "sphysnet_100_aqueous",
                "families": int(sphys100["family_key"].nunique()),
                "states": len(sphys100),
                "scope": "aqueous quantitative logK; training/validation clean; known prior published test exposure",
                "paper_role": "primary_external_confirmation_with_exposure_qualification",
            },
            {
                "dataset": "nci_aqueous_zero_overlap",
                "families": int(nci["family_key"].nunique()) if not nci.empty else 0,
                "states": len(nci),
                "scope": "strict known-source family exclusion; mixed aqueous conditions",
                "paper_role": "secondary_ood_only_due_small_mixed_solvent_subset",
            },
        ]
    )
    external_summary.to_csv(table_dir / "evaluation_external_dataset_manifest.csv", index=False)
    (table_dir / "evaluation_external_dataset_manifest.md").write_text(
        "# evaluation External Dataset Manifest\n\n" + markdown_table(external_summary) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tautobase_clean_families": int(manifest["checkpoint_clean"].sum()),
                "sphysnet_100_families": int(sphys100["family_key"].nunique()),
                "nci_zero_overlap_families": int(nci["family_key"].nunique()) if not nci.empty else 0,
                "nci_sha256": sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
