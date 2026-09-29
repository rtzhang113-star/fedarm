                                                                         

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:                                                                  
    from .engine import run_queue                                  
except (ImportError, ValueError):                                               
    from engine import run_queue                          


DEFAULT_FORMAL_CONFIG = HERE / "rq5_retention_formal_protocol.json"
DEFAULT_SMOKE_CONFIG = HERE / "smoke_config.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="RQ5 JSON config (defaults to the formal prepared protocol)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="select smoke_config.json and run its bounded matrix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without reading caches or writing output",
    )
    parser.add_argument(
        "--run-formal",
        action="store_true",
        help="explicitly execute only a FROZEN, formally authorized 64-slice config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible output, preserving previous cell attempts",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="audit an existing output without writing or acquiring a runner lock",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional RQ5 output override below results/rq5_retention_*",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record a failed cell and continue to the next condition",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.smoke and args.run_formal:
        raise SystemExit("--smoke and --run-formal cannot be combined")
    if args.audit and (args.run_formal or args.resume or args.dry_run or args.smoke):
        raise SystemExit("--audit cannot be combined with an execution action")
    if args.resume and args.dry_run:
        raise SystemExit("--resume and --dry-run cannot be combined")
    if args.continue_on_error and (args.dry_run or args.audit):
        raise SystemExit("--continue-on-error applies only to execution")

    config_path = args.config
    if config_path is None:
        config_path = DEFAULT_SMOKE_CONFIG if args.smoke else DEFAULT_FORMAL_CONFIG
                                                                              
                                                                              
    dry_run = bool(args.dry_run or not any((args.smoke, args.run_formal, args.resume, args.audit)))
    result = run_queue(
        config_path,
        dry_run=dry_run,
        run_formal=bool(args.run_formal),
        resume=bool(args.resume),
        audit=bool(args.audit),
        output=args.output,
        stop_on_error=not bool(args.continue_on_error),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.audit and result.get("status") != "COMPLETE":
        return 2
    if result.get("status") == "INCOMPLETE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
