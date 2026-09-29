                                                                               

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceSplit:
    time_slice: int
    support_users: np.ndarray
    support_services: np.ndarray
    support_values: np.ndarray
    query_users: np.ndarray
    query_services: np.ndarray
    query_values: np.ndarray
    mask_sha256: str

    @property
    def support_count(self) -> int:
        return int(self.support_values.size)

    @property
    def query_count(self) -> int:
        return int(self.query_values.size)


@dataclass(frozen=True)
class CalibrationSplit:
    train_users: np.ndarray
    train_services: np.ndarray
    train_values: np.ndarray
    holdout_users: np.ndarray
    holdout_services: np.ndarray
    holdout_values: np.ndarray
    split_sha256: str


def make_support_calibration_split(
    support_users: np.ndarray,
    support_services: np.ndarray,
    support_values: np.ndarray,
    holdout_fraction: float,
    seed: int,
    time_slice: int,
) -> CalibrationSplit:
                                                                        
    if not (0.0 < holdout_fraction < 0.5):
        raise ValueError("holdout_fraction must lie between zero and 0.5")
    if not (len(support_users) == len(support_services) == len(support_values)):
        raise ValueError("support arrays must have equal length")
    holdout = np.zeros(len(support_values), dtype=bool)
    for user in np.unique(support_users):
        positions = np.flatnonzero(support_users == user)
        if positions.size < 2:
            continue
        count = max(1, int(holdout_fraction * positions.size))
        count = min(count, positions.size - 1)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(time_slice), int(user), 0xCA11B])
        )
        holdout[rng.choice(positions, size=count, replace=False)] = True
    train = ~holdout
    if not train.any() or not holdout.any():
        raise ValueError("calibration split requires non-empty train and holdout sets")
    digest = hashlib.sha256()
    digest.update(np.asarray([seed, time_slice], dtype=np.int64).tobytes())
    digest.update(np.asarray([holdout_fraction], dtype=np.float64).tobytes())
    digest.update(support_users.tobytes())
    digest.update(support_services.tobytes())
    digest.update(np.packbits(holdout).tobytes())
    return CalibrationSplit(
        train_users=support_users[train],
        train_services=support_services[train],
        train_values=support_values[train],
        holdout_users=support_users[holdout],
        holdout_services=support_services[holdout],
        holdout_values=support_values[holdout],
        split_sha256=digest.hexdigest(),
    )


def make_causal_split(values: np.ndarray, density: float, seed: int, time_slice: int) -> SliceSplit:
                                                                                  
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must have shape user x service")
    if not (0.0 < density < 1.0):
        raise ValueError("density must lie strictly between zero and one")
    available = np.isfinite(values) & (values > 0)
    support = np.zeros_like(available)
    for user in range(values.shape[0]):
        services = np.flatnonzero(available[user])
        count = int(density * services.size)
        if count:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(time_slice), int(user)]))
            selected = rng.choice(services, size=count, replace=False)
            support[user, selected] = True
    query = available & ~support
    if np.any(support & query) or np.any(support & ~available):
        raise AssertionError("support/query partition is invalid")
    if int(available.sum()) != int(support.sum() + query.sum()):
        raise AssertionError("support/query counts do not partition available entries")
    support_users, support_services = np.nonzero(support)
    query_users, query_services = np.nonzero(query)
    digest = hashlib.sha256()
    digest.update(np.asarray([seed, time_slice], dtype=np.int64).tobytes())
    digest.update(np.asarray([density], dtype=np.float64).tobytes())
    digest.update(np.packbits(available.ravel()).tobytes())
    digest.update(np.packbits(support.ravel()).tobytes())
    return SliceSplit(
        time_slice=int(time_slice),
        support_users=support_users.astype(np.int64, copy=False),
        support_services=support_services.astype(np.int64, copy=False),
        support_values=values[support_users, support_services].astype(np.float32, copy=False),
        query_users=query_users.astype(np.int64, copy=False),
        query_services=query_services.astype(np.int64, copy=False),
        query_values=values[query_users, query_services].astype(np.float32, copy=False),
        mask_sha256=digest.hexdigest(),
    )
