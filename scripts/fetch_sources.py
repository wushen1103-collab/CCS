"""Fetch upstream inputs on demand and check the recorded input hashes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url):
    req = urllib.request.Request(url,headers={"User-Agent":"CCS-reproduction"})
    with urllib.request.urlopen(req,timeout=120) as response:
        return response.read()


def download(url,path,expected=None):
    content = path.read_bytes() if path.exists() else request(url)
    if expected and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Upstream checksum changed for {path.name}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)


def source_archive(resource,destination):
    if destination.exists():
        print(f"Using existing source directory: {destination.relative_to(ROOT)}")
        return
    repository = resource["url"].removesuffix(".git").removeprefix("https://github.com/")
    metadata = json.loads(request("https://api.github.com/repos/"+repository))
    revision = resource.get("commit") or metadata["default_branch"]
    content = request("https://api.github.com/repos/"+repository+"/zipball/"+revision)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts[1:]
            if not parts or item.is_dir():
                continue
            if ".." in parts or Path(*parts).is_absolute():
                raise ValueError("Unsafe upstream archive path")
            path = destination.joinpath(*parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.read(item))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tautobase",action="store_true")
    parser.add_argument("--experts",action="store_true")
    parser.add_argument("--nci",action="store_true")
    args = parser.parse_args()
    if not any((args.tautobase,args.experts,args.nci)):
        parser.error("select --tautobase, --experts, or --nci")
    resources = json.loads((ROOT/"data/sources.json").read_text())
    for resource in resources:
        name = resource["resource"]
        if name == "Tautobase" and args.tautobase:
            record = next(x for x in resource["files"] if x["path"]=="Tautobase_SMIRKS.txt")
            url = "https://raw.githubusercontent.com/WahlOya/Tautobase/"+resource["commit"]+"/Tautobase_SMIRKS.txt"
            download(url,ROOT/"data/raw/tautobase/Tautobase/Tautobase_SMIRKS.txt",record["sha256"])
        elif name in ("Tautomer-Predictor-src","sPhysNet-Taut-src") and args.experts:
            destination = ROOT/"data/raw/external"/name
            source_archive(resource,destination)
            for record in resource["files"]:
                path = destination/record["path"]
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Obtain the upstream file matching its recorded checksum: {name}/{record['path']}")
        elif name == "NCI Tautomer Database" and args.nci:
            download(resource["url"],ROOT/"data/raw/external/NCI-Tautomer-Database/Tautomer_database_release_3a.xlsx",resource["sha256"])
    print("Selected upstream inputs are present and checksummed.")


if __name__ == "__main__":
    main()
