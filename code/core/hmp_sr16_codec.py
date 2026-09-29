                                                               

                                                                           
                                                                           
                                                                        
                                                                     
   

from __future__ import annotations

import struct
from dataclasses import dataclass
import numpy as np


PRECISION_FP64 = "FP64"
PRECISION_FP32 = "FP32"
PRECISION_INT16 = "INT16"
PRECISION_INT8 = "INT8"
CODEC_FP64 = 0
CODEC_INT16 = 1
CODEC_FP32 = 2
CODEC_INT8 = 3
SCALE_FLOOR = float(2.0 ** -24)
INT16_LIMIT = 32767
INT8_LIMIT = 127
SERVICE_ID_BYTES = 4
CODEC_FLAG_BYTES = 1
FP32_SCALE_BYTES = 4
INT16_VALUE_BYTES = 2
INT8_VALUE_BYTES = 1
FP64_VALUE_BYTES = 8
SUMMARY_COUNT_BYTES = 4


@dataclass(frozen=True)
class DecodedServiceRow:
                                                                     

    service_id: int
    precision: str
    factors: np.ndarray
    bias: float
    error: float
    saturated: bool
    factor_scale: float | None
    scalar_scale: float | None
    proposal_count: int | None = None
    payload_bytes: int = 0


def _precision(value: str) -> str:
    value = str(value).upper()
    if value not in (PRECISION_FP64, PRECISION_FP32, PRECISION_INT16, PRECISION_INT8):
        raise ValueError(f"unsupported HMP precision: {value}")
    return value


def _row_values(factors: np.ndarray, bias: float, error: float, rank: int) -> np.ndarray:
    values = np.asarray(factors, dtype=np.float64)
    if values.shape != (int(rank),) or not np.all(np.isfinite(values)):
        raise ValueError("factors must be finite float64 values with the configured rank")
    scalars = np.asarray([bias, error], dtype=np.float64)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("bias and error must be finite")
    return np.concatenate((values, scalars))


def _quantize(values: np.ndarray, *, limit: int = INT16_LIMIT) -> tuple[np.ndarray, float, bool]:
                                                                               

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("cannot quantize an empty or non-finite group")
    if isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0:
        raise ValueError("quantization limit must be a positive integer")
    limit = int(limit)
    max_abs = float(np.max(np.abs(values)))
                                                                         
                                                                        
    scale = float(np.float32(max(max_abs / limit, SCALE_FLOOR)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError("non-finite quantization scale")
    unrounded = values / scale
    rounded = np.rint(unrounded)
    saturated = bool(np.any(rounded < -limit) or np.any(rounded > limit))
    quantized = np.clip(rounded, -limit, limit)
    dtype = np.int8 if limit == INT8_LIMIT else np.int16
    quantized = quantized.astype(dtype)
    return quantized, scale, saturated


def quantization_saturated(
    factors: np.ndarray,
    bias: float,
    error: float,
    *,
    precision: str,
    rank: int,
) -> bool:
                                                          

                                                                              
                                                                             
                                                                          
       

    precision = _precision(precision)
    values = _row_values(factors, bias, error, int(rank))
    if precision in (PRECISION_FP64, PRECISION_FP32):
        return False
    limit = INT8_LIMIT if precision == PRECISION_INT8 else INT16_LIMIT
    _, _, factor_sat = _quantize(values[:-2], limit=limit)
    _, _, scalar_sat = _quantize(values[-2:], limit=limit)
    return bool(factor_sat or scalar_sat)


def _payload_size(*, precision: str, rank: int, summary: bool) -> int:
    precision = _precision(precision)
    value_bytes = (
        FP64_VALUE_BYTES * (int(rank) + 2)
        if precision == PRECISION_FP64
        else 4 * (int(rank) + 2)
        if precision == PRECISION_FP32
        else FP32_SCALE_BYTES * 2
        + (INT8_VALUE_BYTES if precision == PRECISION_INT8 else INT16_VALUE_BYTES)
        * (int(rank) + 2)
    )
    return SERVICE_ID_BYTES + CODEC_FLAG_BYTES + value_bytes + (
        SUMMARY_COUNT_BYTES if summary else 0
    )


def expected_payload_bytes(precision: str, *, rank: int = 20, summary: bool = False) -> int:
                                                       

    return _payload_size(precision=precision, rank=rank, summary=summary)


def encode_service_row(
    service_id: int,
    factors: np.ndarray,
    bias: float,
    error: float,
    *,
    precision: str,
    rank: int,
    summary_count: int | None = None,
) -> bytes:
                                                                   

    precision = _precision(precision)
    sid = int(service_id)
    if sid < 0 or sid >= 2**32:
        raise ValueError("service_id must fit uint32")
    values = _row_values(factors, bias, error, int(rank))
    is_summary = summary_count is not None
    if is_summary:
        if isinstance(summary_count, bool) or int(summary_count) != summary_count or int(summary_count) < 0:
            raise ValueError("summary_count must be a non-negative integer")
    codec = {
        PRECISION_FP64: CODEC_FP64,
        PRECISION_FP32: CODEC_FP32,
        PRECISION_INT16: CODEC_INT16,
        PRECISION_INT8: CODEC_INT8,
    }[precision]
    payload = bytearray(struct.pack("<IB", sid, codec))
    if precision == PRECISION_FP64:
        payload.extend(struct.pack("<" + "d" * values.size, *values.tolist()))
    elif precision == PRECISION_FP32:
        payload.extend(struct.pack("<" + "f" * values.size, *values.tolist()))
    elif precision == PRECISION_INT16:
        q_factor, factor_scale, _ = _quantize(values[:-2], limit=INT16_LIMIT)
        q_scalar, scalar_scale, _ = _quantize(values[-2:], limit=INT16_LIMIT)
        payload.extend(struct.pack("<ff", factor_scale, scalar_scale))
        payload.extend(struct.pack("<" + "h" * q_factor.size, *q_factor.tolist()))
        payload.extend(struct.pack("<hh", int(q_scalar[0]), int(q_scalar[1])))
    else:
        q_factor, factor_scale, _ = _quantize(values[:-2], limit=INT8_LIMIT)
        q_scalar, scalar_scale, _ = _quantize(values[-2:], limit=INT8_LIMIT)
        payload.extend(struct.pack("<ff", factor_scale, scalar_scale))
        payload.extend(struct.pack("<" + "b" * q_factor.size, *q_factor.tolist()))
        payload.extend(struct.pack("<bb", int(q_scalar[0]), int(q_scalar[1])))
    if is_summary:
        payload.extend(struct.pack("<I", int(summary_count)))
    expected = expected_payload_bytes(precision, rank=int(rank), summary=is_summary)
    if len(payload) != expected:
        raise AssertionError(f"codec payload size mismatch: {len(payload)} != {expected}")
    return bytes(payload)


def decode_service_row(
    payload: bytes | bytearray | memoryview,
    *,
    rank: int,
    summary: bool = False,
) -> DecodedServiceRow:
                                                                            

    raw = bytes(payload)
    if len(raw) < SERVICE_ID_BYTES + CODEC_FLAG_BYTES:
        raise ValueError("truncated HMP service row")
    sid, codec = struct.unpack_from("<IB", raw, 0)
    if codec == CODEC_FP64:
        precision = PRECISION_FP64
    elif codec == CODEC_FP32:
        precision = PRECISION_FP32
    elif codec == CODEC_INT16:
        precision = PRECISION_INT16
    elif codec == CODEC_INT8:
        precision = PRECISION_INT8
    else:
        raise ValueError("unknown HMP codec flag")
    expected = expected_payload_bytes(precision, rank=int(rank), summary=summary)
    if len(raw) != expected:
        raise ValueError(f"unexpected HMP payload length: {len(raw)} != {expected}")
    offset = SERVICE_ID_BYTES + CODEC_FLAG_BYTES
    factor_scale: float | None = None
    scalar_scale: float | None = None
    saturated = False
    if precision == PRECISION_FP64:
        count = int(rank) + 2
        values = np.asarray(struct.unpack_from("<" + "d" * count, raw, offset), dtype=np.float64)
        offset += FP64_VALUE_BYTES * count
    elif precision == PRECISION_FP32:
        count = int(rank) + 2
        values = np.asarray(struct.unpack_from("<" + "f" * count, raw, offset), dtype=np.float64)
        offset += 4 * count
    elif precision == PRECISION_INT16:
        factor_scale, scalar_scale = struct.unpack_from("<ff", raw, offset)
        if not np.isfinite(factor_scale) or not np.isfinite(scalar_scale) or factor_scale <= 0.0 or scalar_scale <= 0.0:
            raise ValueError("invalid HMP INT16 scales")
        offset += FP32_SCALE_BYTES * 2
        factor_q = np.asarray(struct.unpack_from("<" + "h" * int(rank), raw, offset), dtype=np.int16)
        offset += INT16_VALUE_BYTES * int(rank)
        scalar_q = np.asarray(struct.unpack_from("<hh", raw, offset), dtype=np.int16)
        offset += INT16_VALUE_BYTES * 2
        values = np.concatenate(
            (
                factor_q.astype(np.float64) * float(factor_scale),
                scalar_q.astype(np.float64) * float(scalar_scale),
            )
        )
                                                                           
                                                                             
                                                                         
        saturated = False
    else:
        factor_scale, scalar_scale = struct.unpack_from("<ff", raw, offset)
        if not np.isfinite(factor_scale) or not np.isfinite(scalar_scale) or factor_scale <= 0.0 or scalar_scale <= 0.0:
            raise ValueError("invalid HMP INT8 scales")
        offset += FP32_SCALE_BYTES * 2
        factor_q = np.asarray(struct.unpack_from("<" + "b" * int(rank), raw, offset), dtype=np.int8)
        offset += INT8_VALUE_BYTES * int(rank)
        scalar_q = np.asarray(struct.unpack_from("<bb", raw, offset), dtype=np.int8)
        offset += INT8_VALUE_BYTES * 2
        values = np.concatenate(
            (
                factor_q.astype(np.float64) * float(factor_scale),
                scalar_q.astype(np.float64) * float(scalar_scale),
            )
        )
                                                                           
                                                                             
                                                                         
        saturated = False
    proposal_count: int | None = None
    if summary:
        proposal_count = int(struct.unpack_from("<I", raw, offset)[0])
        offset += SUMMARY_COUNT_BYTES
    if offset != len(raw) or not np.all(np.isfinite(values)):
        raise ValueError("invalid HMP decoded row")
    return DecodedServiceRow(
        service_id=int(sid),
        precision=precision,
        factors=np.asarray(values[:-2], dtype=np.float64),
        bias=float(values[-2]),
        error=float(values[-1]),
        saturated=saturated,
        factor_scale=factor_scale,
        scalar_scale=scalar_scale,
        proposal_count=proposal_count,
        payload_bytes=len(raw),
    )


def encode_and_decode_service_row(
    service_id: int,
    factors: np.ndarray,
    bias: float,
    error: float,
    *,
    precision: str,
    rank: int,
    summary_count: int | None = None,
) -> tuple[bytes, DecodedServiceRow]:
                                                                            

    payload = encode_service_row(
        service_id,
        factors,
        bias,
        error,
        precision=precision,
        rank=rank,
        summary_count=summary_count,
    )
    decoded = decode_service_row(payload, rank=rank, summary=summary_count is not None)
    return payload, decoded


__all__ = [
    "CODEC_FP64",
    "CODEC_FP32",
    "CODEC_INT16",
    "CODEC_INT8",
    "DecodedServiceRow",
    "INT8_LIMIT",
    "INT16_LIMIT",
    "PRECISION_FP64",
    "PRECISION_FP32",
    "PRECISION_INT16",
    "PRECISION_INT8",
    "SCALE_FLOOR",
    "decode_service_row",
    "encode_and_decode_service_row",
    "encode_service_row",
    "expected_payload_bytes",
    "quantization_saturated",
]
