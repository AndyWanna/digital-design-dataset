"""Run all available dataset retrievers and log per-retriever success/failure.

Mirrors the logic from build_design_dataset.ipynb but robust to individual
retriever failures (so a network hiccup on one source doesn't kill the rest).
"""

import sys
import traceback
from pathlib import Path

from dotenv import dotenv_values

from digital_design_dataset.data_sources.data_retrievers import (
    EPFLDatasetRetriever,
    HW2VecDatasetRetriever,
    ISCAS85DatasetRetriever,
    ISCAS89DatasetRetriever,
    KoiosDatasetRetriever,
    LGSynth89DatasetRetriever,
    LGSynth91DatasetRetriever,
    OPDBDatasetRetriever,
    OpencoresDatasetRetriever,
    VTRDatasetRetriever,
)
from digital_design_dataset.data_sources.hls_data import PolybenchRetriever
from digital_design_dataset.design_dataset import DesignDataset

here = Path(__file__).parent
env_config = dotenv_values(here / ".env")
gh_token = env_config.get("GITHUB_TOKEN")
if not gh_token:
    print("WARN: no GITHUB_TOKEN found in demo_scripts/.env", file=sys.stderr)

db_dir = here / "test_dataset_v2"
d = DesignDataset(db_dir, overwrite=True, gh_token=gh_token)
print(f"[init] dataset at {db_dir}")

retrievers = [
    ("iscas85", ISCAS85DatasetRetriever),
    ("iscas89", ISCAS89DatasetRetriever),
    ("lgsynth89", LGSynth89DatasetRetriever),
    ("lgsynth91", LGSynth91DatasetRetriever),
    ("vtr", VTRDatasetRetriever),
    ("koios", KoiosDatasetRetriever),
    ("epfl", EPFLDatasetRetriever),
    ("opencores", OpencoresDatasetRetriever),
    ("hw2vec", HW2VecDatasetRetriever),
    ("opdb", OPDBDatasetRetriever),
    ("polybench", PolybenchRetriever),
]

results: dict[str, str] = {}
for name, cls in retrievers:
    print(f"\n[run] {name} ...", flush=True)
    try:
        r = cls(d)
        r.get_dataset()
        results[name] = "OK"
        print(f"[done] {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        results[name] = f"FAIL: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"[fail] {name}: {exc}", flush=True)

print("\n=== SUMMARY ===")
for name, status in results.items():
    print(f"  {name}: {status}")
