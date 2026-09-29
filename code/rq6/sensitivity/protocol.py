                                                                              
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("rq6_sensitivity_formal_protocol.json")
PROTOCOL_ID = "fedarm_rq6_sensitivity_v1"
DEFAULTS = {"rank": 20, "cpsa_gain": 0.1, "count_exponent": 1.0}
AXES = {"rank": [5, 10, 20, 40, 64], "cpsa_gain": [.025, .05, .1, .2, .4], "count_exponent": [0, .5, 1., 1.5, 2.]}
REFERENCE = {"id": "reference", "axis": "reference", "retention_fraction": 1., "zipf_alpha": 0., "participation_rate": 1.}


def value_tag(value: float | int) -> str:
    return format(float(value), ".12g").replace(".", "p")


def validate_output(path: str | Path) -> Path:
    target = Path(path)
    target = (ROOT / target).resolve() if not target.is_absolute() else target.resolve()
    boundary = (ROOT / "results").resolve()
    if not target.is_relative_to(boundary) or target == boundary:
        raise ValueError("RQ6 output must be a child of results")
    if not target.relative_to(boundary).parts[0].startswith("rq6_sensitivity_"):
        raise ValueError("RQ6 batch must have a rq6_sensitivity_ prefix")
    return target


def build_settings(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    settings = [{"setting_id": "default", "axis": "default", "axis_value": None, "parameters": dict(protocol["defaults"])}]
    for axis in protocol["axes"]:
        name = axis["id"]
        for value in axis["values"]:
            if value == protocol["defaults"][name]:
                continue
            parameters = dict(protocol["defaults"])
            parameters[name] = value
            settings.append({"setting_id": f"{name}_{value_tag(value)}", "axis": name, "axis_value": value, "parameters": parameters})
    return settings


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong RQ6 protocol")
    if protocol.get("status") not in {"PREPARED_NOT_AUTHORIZED", "FROZEN"}:
        raise ValueError("unknown preparation/authorization state")
    if type(protocol.get("formal_authorized")) is not bool:
        raise ValueError("formal_authorized must be an explicit boolean")
    if protocol["formal_authorized"] and protocol["status"] != "FROZEN":
        raise ValueError("formal run requires frozen authorization")
    if protocol.get("dataset", {}).get("shape_time_user_service") != [64, 142, 4500]:
        raise ValueError("wrong Dataset 3 shape")
    if protocol.get("base_config") != "configs/base_config.json" or protocol["dataset"].get("cache_dir") != "data/cache":
        raise ValueError("base configuration/data source changed")
    if protocol.get("defaults") != DEFAULTS:
        raise ValueError("default model may not drift")
    if protocol.get("axes") != [{"id": k, "values": v} for k, v in AXES.items()]:
        raise ValueError("sensitivity grid differs from the agreed scope")
                                                                        
    if any(type(v) not in (int, float) for v in protocol["defaults"].values()) or any(type(v) not in (int, float) for a in protocol["axes"] for v in a["values"]):
        raise ValueError("parameters must be numeric, not boolean")
    fixed = {"trust_exponent": .5, "cpsa_l2": .0003, "carw_window": 16, "carw_allocated_columns": 17, "participation_rate": 1., "retention_fraction": 1.}
    if protocol.get("fixed") != fixed:
        raise ValueError("fixed mechanism or safe history scope changed")
    if any(type(v) not in (int, float) for v in protocol["fixed"].values()):
        raise ValueError("fixed numeric fields cannot be boolean")
    expected_matrix = {"densities": [.01, .05, .1, .15, .2], "seeds": [42, 3407, 1024, 1, 0], "qos_types": ["rt", "tp"], "time_slices": 64}
    if protocol.get("matrix") != expected_matrix:
        raise ValueError("formal matrix must remain 650 conditions")
    expected_smoke = {"setting_ids": "all", "densities": [.01], "seeds": [42], "qos_types": ["rt", "tp"], "time_slices": 20}
    if protocol.get("smoke") != expected_smoke:
        raise ValueError("smoke must cover all settings and cross the history boundary")
    for scope in (protocol["matrix"], protocol["smoke"]):
        if any(type(v) is not int for v in scope["seeds"]) or type(scope["time_slices"]) is not int:
            raise ValueError("seeds and horizon must be integers")
    for path in protocol["outputs"].values():
        validate_output(path)
    if validate_output(protocol["outputs"]["smoke"]) == validate_output(protocol["outputs"]["formal"]):
        raise ValueError("smoke and formal outputs must be isolated")
    reuse = protocol["baseline_reuse"]
    if reuse.get("enabled") is not True or reuse.get("require_default_equivalence") is not True:
        raise ValueError("default reuse must be gated, not silently retrained")
    if reuse.get("source_root") != "results/rq4_robustness_formal":
        raise ValueError("only the safe RQ3 reference batch may be reused")
    if validate_output(reuse["equivalence_output"]) != validate_output(protocol["outputs"]["smoke"]):
        raise ValueError("equivalence gate must use this preparation smoke")
    if protocol.get("metrics") != ["mae", "rmse", "mre", "npre"]:
        raise ValueError("all four metrics must be retained")
    if protocol.get("interpretation", {}).get("test_driven_selection") is not False:
        raise ValueError("query cannot select parameters")
    return protocol


def scope_slices(protocol: Mapping[str, Any], smoke: bool = False) -> int:
    return int(protocol["smoke" if smoke else "matrix"]["time_slices"])


def build_cells(protocol: Mapping[str, Any], smoke: bool = False) -> list[dict[str, Any]]:
    scope = protocol["smoke" if smoke else "matrix"]
    cells = []
    for setting in build_settings(protocol):
        for density, seed, qos in itertools.product(scope["densities"], scope["seeds"], scope["qos_types"]):
            cells.append({**setting, "parameters": dict(setting["parameters"]),
                "protocol": PROTOCOL_ID,
                "cell_id": f"{setting['setting_id']}_{qos}_d{value_tag(density)}_seed{seed}",
                "qos": qos, "density": float(density), "seed": int(seed),
                "scenario_id": setting["setting_id"], "scenario": dict(REFERENCE),
                "variant_id": "R0", "variant": "full",
                "execution_kind": "REUSE" if setting["setting_id"] == "default" and not smoke else "TRAIN"})
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("duplicate cell identity")
    return cells


def axis_memberships(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
                                                                          
    if cell["setting_id"] == "default":
        return [{"axis": name, "axis_value": DEFAULTS[name]} for name in AXES]
    return [{"axis": cell["axis"], "axis_value": cell["axis_value"]}]
