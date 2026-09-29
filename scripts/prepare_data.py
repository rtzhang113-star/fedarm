                                                              

                                                                            
                                                             
                                                                              
                                               
   

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continual_qos.data import build_value_cache              


DEFAULT_RAW_DIR = ROOT / "data" / "WSDream-QoSDataset3"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
QOS_TYPES = ("rt", "tp")
RAW_NAMES = {"rt": "rtRate", "tp": "tpRate"}


def _project_path(value: Path) -> Path:
                                                                               

    return value if value.is_absolute() else value.resolve()


def _raw_path(raw_dir: Path, qos_type: str) -> Path:
                                                                               

    stem = RAW_NAMES[qos_type]
    for candidate in (raw_dir / stem, raw_dir / f"{stem}.txt", raw_dir / f"{stem}.csv"):
        if candidate.is_file():
            return candidate
    return raw_dir / stem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qos",
        nargs="+",
        choices=QOS_TYPES,
        default=list(QOS_TYPES),
        help="QoS files to convert (default: rt tp).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing rtRate and tpRate (default: data/WSDream-QoSDataset3).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Destination for .npy caches and manifests (default: data/cache).",
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("TIME", "USER", "SERVICE"),
        default=(64, 142, 4500),
        help="Cache shape (default: 64 142 4500).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    shape = tuple(int(value) for value in args.shape)
    if any(value <= 0 for value in shape):
        raise ValueError("all shape dimensions must be positive")
    raw_dir = _project_path(args.raw_dir)
    cache_dir = _project_path(args.cache_dir)
    manifests: dict[str, dict] = {}
    for qos_type in dict.fromkeys(args.qos):
        raw_path = _raw_path(raw_dir, qos_type)
        manifests[qos_type] = build_value_cache(
            raw_path=raw_path,
            cache_path=cache_dir / f"{qos_type}_values.npy",
            manifest_path=cache_dir / f"{qos_type}_data_manifest.json",
            qos_type=qos_type,
            shape=shape,
        )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
