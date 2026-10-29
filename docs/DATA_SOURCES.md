# Data and external experts

`data/sources.json` records original public sources and critical input hashes.
The repository includes structure-free numerical predictions and fixed split
assignments. Original structures, licensed crystallographic training collections,
third-party source trees and expert weights are not redistributed here.

| Resource | Upstream source | Use |
|---|---|---|
| Tautobase | https://github.com/WahlOya/Tautobase | Curated two-state preferences and fixed family split. The recorded source revision and raw-file checksum are checked during download. |
| Tautomer-Predictor | https://github.com/xiaolinpan/Tautomer-Predictor | Released 2D AttentiveFP expert. Source code is MIT-licensed. The recorded checkpoint is approximately 67 MB and is downloaded only for fresh expert inference. |
| sPhysNet-Taut | https://github.com/xiaolinpan/sPhysNet-Taut | Released energy expert. Its GPL source license remains upstream. Aqueous confirmation uses published model-predicted energies; local energy inference uses ten requested conformers. |
| NCI Tautomer Database | https://cactus.nci.nih.gov/download/tautomer/ | Release 3a secondary transfer pool with aqueous eligibility, duplicate handling and known-source exclusion. Obtain the original XLSX directly under the source's access terms. |

Tautobase labels retain their original mixed-condition, mixed-origin preference
annotations. Missing source-type flags are not converted to measured labels.
For aqueous confirmation, experimental free-energy signs define the outcomes,
and the released model-predicted differences define the energy-expert inputs.
Prediction and experimental values serve separate roles.

The historical expert source directories did not retain separate Git metadata.
Their critical checkpoint, released-data and license-file hashes are recorded;
the fetch command verifies these files against the upstream download. If an
upstream resource changes, obtain the file matching the recorded checksum.

NCI curation begins with 149 candidate records and retains 28 chemically eligible
families. Six fail the energy expert's element vocabulary, leaving 22 families
supported by all configurations. Duplicate flags are assigned in original
candidate order before joint eligibility filtering. The 76-family Tautobase
subset excludes accessible-source intersections; unavailable CSD training
structures are recorded as unavailable in the audit, not as zero overlap.
