                                                                       

                                                                            
                                                                             
                                                                              
   

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("rq4_robustness_formal_protocol.json")
VARIANTS = {"R0": "full", "R1": "service_mean_matched"}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str, lower: float, upper: float, *, open_lower: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result > upper or result < lower or (open_lower and result == lower):
        raise ValueError(f"invalid {name}: {value}")
    return result


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(scenario.get("id", ""))):
        raise ValueError("scenario id must be a safe lowercase identifier")
    if scenario.get("axis") not in {"reference", "imbalance", "participation", "joint"}:
        raise ValueError("unknown scenario axis")
    _number(scenario.get("retention_fraction"), "retention_fraction", 0, 1, open_lower=True)
    _number(scenario.get("zipf_alpha"), "zipf_alpha", 0, 8)
    _number(scenario.get("participation_rate"), "participation_rate", 0, 1, open_lower=True)
    if scenario["retention_fraction"] == 1 and scenario["zipf_alpha"] != 0:
        raise ValueError("imbalance requires thinning: alpha has no effect at retention=1")
    retention, alpha, participation = (float(scenario[k]) for k in
                                       ("retention_fraction", "zipf_alpha", "participation_rate"))
    axis = scenario["axis"]
    if axis == "reference" and (retention, alpha, participation) != (1.0, 0.0, 1.0):
        raise ValueError("reference must have no intervention")
    if axis == "imbalance" and not (retention < 1 and participation == 1):
        raise ValueError("imbalance requires a thinned pool and full participation")
    if axis == "participation" and not (retention == 1 and alpha == 0 and participation < 1):
        raise ValueError("participation axis must change only client participation")
    if axis == "joint" and not (retention < 1 and participation < 1):
        raise ValueError("joint axis requires thinning and partial participation")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if protocol.get("protocol_id") != "fedarm_rq4_robustness_v1":
        raise ValueError("wrong RQ4 protocol")
    if protocol.get("status") not in {"PREPARED_NOT_AUTHORIZED", "FROZEN"}:
        raise ValueError("unknown protocol status")
    if type(protocol.get("formal_authorized")) is not bool:
        raise ValueError("formal_authorized must be an explicit boolean")
    if protocol["formal_authorized"] and protocol["status"] != "FROZEN":
        raise ValueError("formal authorization requires a frozen protocol")
    if protocol.get("dataset", {}).get("shape_time_user_service") != [64, 142, 4500]:
        raise ValueError("Dataset 3 shape must remain 64 x 142 x 4500")
    scenarios = protocol["scenarios"]
    if not scenarios:
        raise ValueError("empty scenarios")
    for scenario in scenarios:
        validate_scenario(scenario)
    if len({item["id"] for item in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ids")
    conditions = [(s["retention_fraction"], s["zipf_alpha"], s["participation_rate"]) for s in scenarios]
    if len(set(conditions)) != len(conditions):
        raise ValueError("duplicate intervention conditions")
    variants = protocol["variants"]
    if {(v["variant_id"], v["variant"]) for v in variants} != set(VARIANTS.items()) or len(variants) != 2:
        raise ValueError("RQ4 requires the two matched aggregation variants R0 and R1")
    for scope in (protocol["matrix"], protocol["smoke"]):
        if not scope["densities"] or not scope["seeds"] or not scope["qos_types"]:
            raise ValueError("empty matrix dimension")
        for density in scope["densities"]:
            _number(density, "density", 0, 1, open_lower=True)
            if density >= 1:
                raise ValueError("density must be below 1")
        for seed in scope["seeds"]:
            _integer(seed, "seed")
        if not set(scope["qos_types"]).issubset({"rt", "tp"}):
            raise ValueError("unknown QoS type")
        if not 1 <= _integer(scope["time_slices"], "time_slices", 1) <= 64:
            raise ValueError("time_slices exceeds Dataset 3")
        for dimension in ("densities", "seeds", "qos_types"):
            if len(set(scope[dimension])) != len(scope[dimension]):
                raise ValueError(f"duplicate {dimension}")
    if protocol["matrix"]["time_slices"] != 64:
        raise ValueError("formal matrix requires 64 slices")
    if not set(protocol["smoke"]["scenario_ids"]).issubset({s["id"] for s in scenarios}):
        raise ValueError("unknown smoke scenario")
    if not protocol["smoke"]["scenario_ids"] or len(set(protocol["smoke"]["scenario_ids"])) != len(protocol["smoke"]["scenario_ids"]):
        raise ValueError("empty or duplicate smoke scenario ids")
    return protocol


def scope_slices(protocol: Mapping[str, Any], smoke: bool = False) -> int:
    return int(protocol["smoke" if smoke else "matrix"]["time_slices"])


def build_cells(protocol: Mapping[str, Any], smoke: bool = False) -> list[dict[str, Any]]:
    scope = protocol["smoke" if smoke else "matrix"]
    scenarios = [s for s in protocol["scenarios"] if not smoke or s["id"] in scope["scenario_ids"]]
    cells = []
    for scenario in scenarios:
        for density in scope["densities"]:
            density_tag = format(float(density), ".12g").replace(".", "p")
            for seed in scope["seeds"]:
                for qos in scope["qos_types"]:
                    for variant in protocol["variants"]:
                        cells.append({
                            "cell_id": f"{scenario['id']}_{variant['variant_id']}_{qos}_d{density_tag}_seed{seed}",
                            "scenario_id": scenario["id"], "scenario": dict(scenario),
                            "variant_id": variant["variant_id"], "variant": variant["variant"],
                            "qos": qos, "density": float(density), "seed": int(seed),
                        })
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("cell identity collision")
    return cells


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _gini(counts: np.ndarray) -> float:
    ordered = np.sort(np.asarray(counts, dtype=np.float64))
    total = float(ordered.sum())
    if total == 0:
        return 0.0
    size = ordered.size
    return float((2 * np.dot(np.arange(1, size + 1), ordered) / total - size - 1) / size)


@dataclass(frozen=True)
class Intervention:
    indices: np.ndarray
    selected_clients: np.ndarray
    observed_clients: np.ndarray
    service_counts: np.ndarray
    audit: dict[str, Any]


def intervene_support(
    users: np.ndarray, services: np.ndarray, *, num_users: int, num_services: int,
    seed: int, time_slice: int, scenario: Mapping[str, Any],
) -> Intervention:
                                                                         

                                                                       
                                                                        
                                                                        
                                                                            
                                                                             
       
    validate_scenario(scenario)
    num_users = _integer(num_users, "num_users", 1)
    num_services = _integer(num_services, "num_services", 1)
    seed = _integer(seed, "seed")
    time_slice = _integer(time_slice, "time_slice")
    users, services = np.asarray(users), np.asarray(services)
    if users.ndim != 1 or services.ndim != 1 or users.shape != services.shape:
        raise ValueError("support key arrays must be equal one-dimensional arrays")
    if users.dtype.kind not in "iu" or services.dtype.kind not in "iu":
        raise ValueError("support keys must be integer arrays")
    if np.any(users < 0) or np.any(users >= num_users) or np.any(services < 0) or np.any(services >= num_services):
        raise ValueError("support key out of range")
    users, services = users.astype(np.int64, copy=False), services.astype(np.int64, copy=False)
    keys = users * num_services + services
    if np.unique(keys).size != keys.size:
        raise ValueError("duplicate user-service support key")

    rank_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5E471CE]))
    service_ranks = np.empty(num_services, dtype=np.int64)
    service_ranks[rank_rng.permutation(num_services)] = np.arange(1, num_services + 1)
    alpha = float(scenario["zipf_alpha"])
    retained = np.zeros(users.size, dtype=bool)
    fraction = float(scenario["retention_fraction"])
    for user in np.unique(users):
        positions = np.flatnonzero(users == user)
                                                                           
                                                                          
        positions = positions[np.argsort(services[positions], kind="stable")]
        budget = int(math.floor(fraction * positions.size))
        if budget == positions.size:
            retained[positions] = True
        elif budget > 0:
            rng = np.random.default_rng(np.random.SeedSequence([seed, time_slice, int(user), 0x1B41]))
            uniform = np.clip(rng.random(positions.size), np.finfo(np.float64).tiny, 1 - np.finfo(np.float64).eps)
            scores = -alpha * np.log(service_ranks[services[positions]]) - np.log(-np.log(uniform))
            chosen = np.argsort(-scores, kind="stable")[:budget]
            retained[positions[chosen]] = True
    participation_rng = np.random.default_rng(np.random.SeedSequence([seed, time_slice, 0xC11E17]))
    selected_count = int(math.ceil(float(scenario["participation_rate"]) * num_users))
    selected = np.sort(participation_rng.permutation(num_users)[:selected_count]).astype(np.int64)
    participating = np.zeros(num_users, dtype=bool)
    participating[selected] = True
    indices = np.flatnonzero(retained & participating[users]).astype(np.int64)
    observed = np.unique(users[indices]).astype(np.int64)
    counts = np.bincount(services[indices], minlength=num_services).astype(np.int64)
    audit = {
        "base_support_count": int(users.size),
        "retained_support_count": int(retained.sum()),
        "effective_support_count": int(indices.size),
        "withheld_support_count": int(users.size - indices.size),
        "selected_client_count": int(selected.size),
        "observed_client_count": int(observed.size),
        "realized_participation_rate": float(selected.size / num_users),
        "retention_fraction": fraction, "zipf_alpha": alpha,
        "participation_rate": float(scenario["participation_rate"]),
        "service_support_gini": _gini(counts),
        "service_support_cv": float(counts.std() / counts.mean()) if indices.size else 0.0,
        "zero_support_services": int(np.count_nonzero(counts == 0)),
        "single_support_services": int(np.count_nonzero(counts == 1)),
        "max_service_support": int(counts.max()),
        "base_support_sha256": _hash_arrays(users, services),
        "effective_support_sha256": _hash_arrays(users[indices], services[indices]),
        "participation_sha256": _hash_arrays(selected),
        "retained_support_sha256": _hash_arrays(users[retained], services[retained]),
        "service_counts_sha256": _hash_arrays(counts),
        "service_rank_sha256": _hash_arrays(service_ranks),
        "selected_clients": selected.tolist(),
        "observed_clients": observed.tolist(),
        "base_client_support_counts": np.bincount(users, minlength=num_users).tolist(),
        "retained_client_support_counts": np.bincount(users[retained], minlength=num_users).tolist(),
        "effective_client_support_counts": np.bincount(users[indices], minlength=num_users).tolist(),
        "selection_uses_qos_values": False,
    }
    for array in (indices, selected, observed, counts):
        array.setflags(write=False)
    return Intervention(indices, selected, observed, counts, audit)
