"""Phase 2: detect purely combinatorial, self-contained Verilog files across
the retrieved design dataset and copy them flat into roverlite/designs/.

For each .v/.sv file in every design:
  1. Run yosys: read_verilog -sv; proc; stat -json
  2. Qualify as IDEAL iff:
       - no $dff*/$dlatch*/$sr*/$adff*/$sdff*/$dffe* cells (no clocked or
         latched logic after `proc`)
       - exactly one module defined in the file
       - zero submodule (user-defined) instantiations
  3. IDEAL files are copied to:
       roverlite/designs/<dataset>/<design_short>.v
     where <design_short> = design dir name with the leading "<dataset>__" stripped.
     If a design has >1 ideal file, the filestem is appended:
       roverlite/designs/<dataset>/<design_short>__<filestem>.v

The existing roverlite/designs/adder.sv is left untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

YOSYS_BIN = "/usr/scratch/awanna3/yosys-install/oss-cad-suite/bin/yosys"
OSS_CAD_BIN = "/usr/scratch/awanna3/yosys-install/oss-cad-suite/bin"
os.environ["PATH"] = OSS_CAD_BIN + os.pathsep + os.environ.get("PATH", "")

REPO_ROOT = Path("/usr/scratch/awanna3/EgraphRL")
DATASET_ROOT = REPO_ROOT / "digital-design-dataset/demo_scripts/test_dataset_v2/designs"
ROVERLITE_DESIGNS = REPO_ROOT / "egraphrl-rs/src/bin/roverlite/designs"

# Cell types that indicate sequential / latched logic (post-`proc`).
SEQ_CELL_PREFIXES = (
    "$dff", "$adff", "$sdff", "$dffe", "$adffe", "$sdffe",
    "$aldff", "$aldffe", "$dffsr", "$dffsre",
    "$dlatch", "$adlatch", "$dlatchsr",
    "$sr", "$_DFF", "$_DFFE", "$_SDFF", "$_DLATCH", "$_SR_",
    "$_ALDFF", "$_DFFSR",
)


@dataclass
class FileReport:
    design_dir: str
    file: str
    status: str  # IDEAL | COMB_WITH_SUBMODS | MULTI_MODULE | SEQUENTIAL | READ_FAIL | TIMEOUT | OTHER
    module_count: int = 0
    seq_cells: int = 0
    submod_instances: int = 0
    modules: list[str] = field(default_factory=list)
    note: str = ""


def run_yosys(src: Path, json_out: Path) -> tuple[int, str, str]:
    cmds = f"read_verilog -sv {src}; proc; write_json {json_out}"
    p = subprocess.run(
        [YOSYS_BIN, "-q", "-p", cmds],
        capture_output=True, text=True, timeout=120,
    )
    return p.returncode, p.stdout, p.stderr


def classify_file(design_dir: Path, src: Path) -> FileReport:
    rep = FileReport(design_dir=design_dir.name, file=src.name, status="OTHER")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        json_path = Path(tf.name)
    try:
        try:
            rc, out, err = run_yosys(src, json_path)
        except subprocess.TimeoutExpired:
            rep.status = "TIMEOUT"
            return rep
        if rc != 0 or not json_path.exists() or json_path.stat().st_size == 0:
            rep.status = "READ_FAIL"
            tail = (err or out).strip().splitlines()[-1] if (err or out).strip() else ""
            rep.note = tail[:200]
            return rep
        try:
            with json_path.open() as fh:
                data = json.load(fh)
        except Exception as exc:
            rep.status = "OTHER"
            rep.note = f"json-parse-fail: {exc}"
            return rep
    finally:
        try:
            json_path.unlink()
        except OSError:
            pass

    modules = data.get("modules", {}) or {}
    # Filter out blackbox/auto-generated modules (names starting with '$').
    module_names = [m for m in modules if not m.startswith("$")]
    rep.modules = module_names
    rep.module_count = len(module_names)

    seq = 0
    user_inst = 0
    for _mname, info in modules.items():
        # write_json emits per-cell dicts with 'type' field; aggregate.
        for cell in (info.get("cells", {}) or {}).values():
            ctype = cell.get("type", "")
            if any(ctype.startswith(p) for p in SEQ_CELL_PREFIXES):
                seq += 1
            elif not ctype.startswith("$"):
                # Any user-defined (or library) module instantiation. We
                # conservatively count all non-builtin types as "nested" —
                # includes ISCAS89's FD1/ND2/AN2 library gates and any
                # genuine submodule.
                user_inst += 1

    rep.seq_cells = seq
    rep.submod_instances = user_inst

    if seq > 0:
        rep.status = "SEQUENTIAL"
    elif rep.module_count != 1:
        rep.status = "MULTI_MODULE" if rep.module_count > 1 else "OTHER"
    elif user_inst > 0:
        rep.status = "COMB_WITH_SUBMODS"
    else:
        rep.status = "IDEAL"
    return rep


def gather_tasks() -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for design_dir in sorted(DATASET_ROOT.iterdir()):
        src_dir = design_dir / "sources"
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in (".v", ".sv"):
                tasks.append((design_dir, f))
    return tasks


def split_dataset_design(design_dir_name: str) -> tuple[str, str]:
    # design_dir names are like "epfl__adder" or
    # "hls_polybench__fixed__medium__2mm" -> first token before "__" is dataset.
    prefix, _, rest = design_dir_name.partition("__")
    return prefix, rest or design_dir_name


def main() -> int:
    if not YOSYS_BIN or not Path(YOSYS_BIN).exists():
        print(f"yosys not found at {YOSYS_BIN}", file=sys.stderr)
        return 1

    tasks = gather_tasks()
    print(f"[phase2] classifying {len(tasks)} files across {len({t[0] for t in tasks})} designs")

    reports: list[FileReport] = []
    n_jobs = max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(classify_file, d, f): (d, f) for d, f in tasks}
        done = 0
        for fut in as_completed(futs):
            rep = fut.result()
            reports.append(rep)
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(tasks)}", flush=True)

    # Group IDEAL results by design.
    by_design: dict[str, list[FileReport]] = {}
    for r in reports:
        if r.status == "IDEAL":
            by_design.setdefault(r.design_dir, []).append(r)

    # Copy to roverlite/designs/<dataset>/<design_short>[_filestem].v
    copied = 0
    per_dataset_copied: dict[str, int] = {}
    for design_name, ideal_files in sorted(by_design.items()):
        dataset, design_short = split_dataset_design(design_name)
        out_dir = ROVERLITE_DESIGNS / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        multi = len(ideal_files) > 1
        for r in ideal_files:
            src = DATASET_ROOT / r.design_dir / "sources" / r.file
            stem = Path(r.file).stem
            if multi:
                out_name = f"{design_short}__{stem}.v"
            else:
                out_name = f"{design_short}.v"
            dst = out_dir / out_name
            shutil.copy2(src, dst)
            copied += 1
            per_dataset_copied[dataset] = per_dataset_copied.get(dataset, 0) + 1

    # Summary.
    status_counts: dict[str, int] = {}
    for r in reports:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    print("\n=== FILE-LEVEL STATUS ===")
    for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s} {n}")
    print(f"\n=== COPIED: {copied} files into roverlite/designs/ ===")
    for ds, n in sorted(per_dataset_copied.items(), key=lambda x: -x[1]):
        print(f"  {ds:20s} {n}")
    n_designs_with_ideal = len(by_design)
    print(f"\n=== DESIGNS WITH ≥1 IDEAL FILE: {n_designs_with_ideal} / {len({t[0].name for t in tasks})} ===")

    # Write JSON report for debugging / future use.
    report_path = REPO_ROOT / "digital-design-dataset/demo_scripts/phase2_report.json"
    with report_path.open("w") as fh:
        json.dump(
            {
                "status_counts": status_counts,
                "copied": copied,
                "per_dataset_copied": per_dataset_copied,
                "files": [
                    {
                        "design": r.design_dir,
                        "file": r.file,
                        "status": r.status,
                        "modules": r.modules,
                        "seq_cells": r.seq_cells,
                        "submod_instances": r.submod_instances,
                        "note": r.note,
                    }
                    for r in reports
                ],
            },
            fh, indent=2,
        )
    print(f"\n[phase2] full report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
