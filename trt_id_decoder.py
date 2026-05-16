"""Opt-in TensorRT wrapper for MOTIP trajectory-modeling + ID decoder."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any
import warnings

import torch
from torch import nn


INPUT_NAMES = (
    "trajectory_features",
    "trajectory_id_labels",
    "trajectory_times",
    "trajectory_masks",
    "unknown_features",
    "unknown_masks",
    "unknown_times",
)

_ENGINE_SHAPE_RE = re.compile(r"_t(?P<history>\d+)_n(?P<tracks>\d+)_m(?P<detections>\d+)\.plan$")


def _get_trt_logger(trt):
    logger = getattr(trt, "_motip_logger", None)
    if logger is None:
        logger = trt.Logger(trt.Logger.ERROR)
        setattr(trt, "_motip_logger", logger)
    return logger


class _IdDecoderExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.trajectory_modeling = model.trajectory_modeling
        self.id_decoder = model.id_decoder

    def forward(
        self,
        trajectory_features: torch.Tensor,
        trajectory_id_labels: torch.Tensor,
        trajectory_times: torch.Tensor,
        trajectory_masks: torch.Tensor,
        unknown_features: torch.Tensor,
        unknown_masks: torch.Tensor,
        unknown_times: torch.Tensor,
    ) -> torch.Tensor:
        seq_info = {
            "trajectory_features": trajectory_features,
            "trajectory_id_labels": trajectory_id_labels,
            "trajectory_times": trajectory_times,
            "trajectory_masks": trajectory_masks,
            "unknown_features": unknown_features,
            "unknown_masks": unknown_masks,
            "unknown_times": unknown_times,
        }
        seq_info = self.trajectory_modeling(seq_info)
        id_logits, _, _ = self.id_decoder(seq_info, use_decoder_checkpoint=False)
        return id_logits


def _engine_shape_from_path(engine_path: Path) -> tuple[int, int, int]:
    match = _ENGINE_SHAPE_RE.search(engine_path.name)
    if match is None:
        raise ValueError(
            f"MOTIP ID TensorRT engine filename must include _t<history>_n<tracks>_m<detections>: {engine_path}"
        )
    return int(match.group("history")), int(match.group("tracks")), int(match.group("detections"))


def _make_id_decoder_inputs(
    history: int,
    tracks: int,
    detections: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    trajectory_features = torch.zeros((1, 1, history, tracks, 256), dtype=dtype, device=device)
    trajectory_id_labels = torch.zeros((1, 1, history, tracks), dtype=torch.int64, device=device)
    trajectory_times = torch.arange(history, dtype=torch.int64, device=device).view(1, 1, history, 1).expand(1, 1, history, tracks)
    trajectory_masks = torch.zeros((1, 1, history, tracks), dtype=torch.bool, device=device)
    unknown_features = torch.zeros((1, 1, 1, detections, 256), dtype=dtype, device=device)
    unknown_masks = torch.zeros((1, 1, 1, detections), dtype=torch.bool, device=device)
    unknown_times = torch.full((1, 1, 1, detections), history, dtype=torch.int64, device=device)
    return (
        trajectory_features,
        trajectory_id_labels,
        trajectory_times,
        trajectory_masks,
        unknown_features,
        unknown_masks,
        unknown_times,
    )


def _build_engine_from_onnx(onnx_path: Path, engine_path: Path, dtype: torch.dtype) -> None:
    import tensorrt as trt

    builder = trt.Builder(_get_trt_logger(trt))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, _get_trt_logger(trt))
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse MOTIP ID decoder ONNX for TensorRT:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build MOTIP ID decoder TensorRT engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def ensure_trt_id_engines(model: nn.Module, engine_paths: list[Path], device: torch.device,
                          dtype: torch.dtype = torch.float32) -> list[Path]:
    wrapper = None
    ensured_paths = []
    for raw_path in engine_paths:
        engine_path = Path(raw_path).expanduser().resolve()
        ensured_paths.append(engine_path)
        if engine_path.exists():
            continue
        history, tracks, detections = _engine_shape_from_path(engine_path)
        if wrapper is None:
            wrapper = _IdDecoderExportWrapper(model).eval()
        inputs = _make_id_decoder_inputs(history, tracks, detections, dtype, device)
        onnx_path = engine_path.with_suffix(".onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message="Exporting aten::index operator.*", category=UserWarning)
            warnings.filterwarnings("ignore", message="Constant folding - Only steps=1.*", category=UserWarning)
            torch.onnx.export(
                wrapper,
                inputs,
                str(onnx_path),
                input_names=list(INPUT_NAMES),
                output_names=["id_logits"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        _build_engine_from_onnx(onnx_path, engine_path, dtype)
    return ensured_paths


class _TensorRTIdEngine:
    def __init__(self, engine_path: Path, device: torch.device):
        import tensorrt as trt

        self.engine_path = Path(engine_path).expanduser().resolve()
        self.device = device
        self.trt = trt
        runtime = trt.Runtime(_get_trt_logger(trt))
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT ID decoder engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT ID decoder execution context")
        self.stream = torch.cuda.Stream(device=device)
        self.input_specs = self._read_input_specs()
        trajectory_shape = self.input_specs["trajectory_features"][0]
        unknown_shape = self.input_specs["unknown_features"][0]
        self.history = trajectory_shape[2]
        self.max_tracks = trajectory_shape[3]
        self.max_detections = unknown_shape[3]

    def _read_input_specs(self) -> dict[str, tuple[tuple[int, ...], Any]]:
        specs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                specs[name] = (tuple(self.engine.get_tensor_shape(name)), self.engine.get_tensor_dtype(name))
        if set(specs) != set(INPUT_NAMES):
            raise RuntimeError(f"Unexpected TensorRT ID decoder inputs: {sorted(specs)}")
        return specs

    def _torch_dtype_for_trt(self, dtype: Any) -> torch.dtype:
        mapping = {
            self.trt.DataType.FLOAT: torch.float32,
            self.trt.DataType.HALF: torch.float16,
            self.trt.DataType.INT32: torch.int32,
            self.trt.DataType.INT64: torch.int64,
            self.trt.DataType.BOOL: torch.bool,
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
        return mapping[dtype]

    def fits(self, history: int, tracks: int, detections: int) -> bool:
        return history <= self.history and tracks <= self.max_tracks and detections <= self.max_detections

    def _padded_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        expected_shape, expected_dtype = self.input_specs[name]
        expected_torch_dtype = self._torch_dtype_for_trt(expected_dtype)
        if any(actual > expected for actual, expected in zip(tensor.shape, expected_shape)):
            raise RuntimeError(f"TensorRT ID decoder input {name} expects <= {expected_shape}, got {tuple(tensor.shape)}")

        if tensor.device != self.device or tensor.dtype != expected_torch_dtype:
            tensor = tensor.to(device=self.device, dtype=expected_torch_dtype)
        tensor = tensor.contiguous()
        if tuple(tensor.shape) == expected_shape:
            return tensor

        fill_value = True if expected_torch_dtype == torch.bool else 0
        padded = torch.full(expected_shape, fill_value, device=self.device, dtype=expected_torch_dtype)
        slices = tuple(slice(0, size) for size in tensor.shape)
        padded[slices] = tensor
        return padded.contiguous()

    def _make_outputs(self) -> dict[str, torch.Tensor]:
        outputs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.OUTPUT:
                continue
            outputs[name] = torch.empty(
                tuple(self.engine.get_tensor_shape(name)),
                device=self.device,
                dtype=self._torch_dtype_for_trt(self.engine.get_tensor_dtype(name)),
            )
        return outputs

    @torch.no_grad()
    def run(self, seq_info: dict[str, torch.Tensor]) -> torch.Tensor:
        bindings = {name: self._padded_input(name, seq_info[name]) for name in INPUT_NAMES}
        outputs = self._make_outputs()
        bindings.update(outputs)
        for name, tensor in bindings.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Failed to bind TensorRT ID decoder tensor address: {name}")

        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT ID decoder execute_async_v3 returned False")
        for tensor in outputs.values():
            tensor.record_stream(self.stream)
        current_stream.wait_stream(self.stream)
        return outputs["id_logits"]


class TensorRTIdPredictor:
    def __init__(self, engine_paths: list[Path], device: torch.device, fallback_conf_margin: float = 0.0):
        if not engine_paths:
            raise ValueError("At least one TensorRT ID decoder engine is required.")
        self.device = device
        self.fallback_conf_margin = float(fallback_conf_margin)
        self.engines = sorted(
            (_TensorRTIdEngine(path, device) for path in engine_paths),
            key=lambda engine: (engine.max_detections, engine.max_tracks, engine.history),
        )
        self.tier_calls = defaultdict(int)
        self.fallback_calls = 0
        self.ambiguity_fallback_calls = 0

    def select_engine(self, history: int, tracks: int, detections: int) -> _TensorRTIdEngine | None:
        for engine in self.engines:
            if engine.fits(history, tracks, detections):
                return engine
        return None

    def predict_logits(self, seq_info: dict[str, torch.Tensor]) -> torch.Tensor | None:
        history = seq_info["trajectory_features"].shape[2]
        tracks = seq_info["trajectory_features"].shape[3]
        detections = seq_info["unknown_features"].shape[3]
        if detections == 0:
            first_engine = self.engines[0]
            return torch.empty(
                (1, 1, 1, 0, 51),
                device=self.device,
                dtype=first_engine._torch_dtype_for_trt(first_engine.engine.get_tensor_dtype("id_logits")),
            )
        engine = self.select_engine(history, tracks, detections)
        if engine is None:
            self.fallback_calls += 1
            return None
        logits = engine.run(seq_info)[..., :detections, :]
        self.tier_calls[engine.max_detections] += 1
        return logits

    def summary(self) -> str:
        tier_text = ", ".join(f"M{tier}:{count}" for tier, count in sorted(self.tier_calls.items()))
        if not tier_text:
            tier_text = "none"
        return f"TensorRT ID decoder calls: {tier_text}; fallbacks:{self.fallback_calls}; ambiguity_fallbacks:{self.ambiguity_fallback_calls}"


def _has_assignment_ambiguity(id_scores: torch.Tensor, runtime_tracker: Any, margin: float) -> bool:
    if margin <= 0 or id_scores.numel() == 0:
        return False
    trajectory_id_labels_set = set(runtime_tracker.trajectory_id_labels[0].tolist())
    object_max_labels = set(torch.max(id_scores, dim=-1).indices.tolist())
    valid_columns = [
        label for label in trajectory_id_labels_set & object_max_labels
        if 0 <= label < id_scores.shape[1]
    ]
    if not valid_columns:
        return False

    valid_scores = id_scores[:, valid_columns]
    if valid_scores.shape[0] >= 2:
        id_top_values = torch.topk(valid_scores, k=2, dim=0).values
        if bool(((id_top_values[0] - id_top_values[1]).abs() <= margin).any().item()):
            return True
    return False


def install_trt_id_predictor(runtime_tracker: Any, predictor: TensorRTIdPredictor) -> None:
    original_get_id_pred_labels = runtime_tracker._get_id_pred_labels

    @torch.no_grad()
    def _get_id_pred_labels(boxes: torch.Tensor, output_embeds: torch.Tensor) -> torch.Tensor:
        if runtime_tracker.trajectory_features.shape[0] == 0:
            return runtime_tracker.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)

        current_features = output_embeds[None, ...]
        current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=boxes.device)
        current_times = runtime_tracker.trajectory_times.shape[0] * torch.ones(
            (1, output_embeds.shape[0]), dtype=torch.int64, device=boxes.device
        )
        seq_info = {
            "trajectory_features": runtime_tracker.trajectory_features[None, None, ...],
            "trajectory_id_labels": runtime_tracker.trajectory_id_labels[None, None, ...],
            "trajectory_times": runtime_tracker.trajectory_times[None, None, ...],
            "trajectory_masks": runtime_tracker.trajectory_masks[None, None, ...],
            "unknown_features": current_features[None, None, ...],
            "unknown_masks": current_masks[None, None, ...],
            "unknown_times": current_times[None, None, ...],
        }
        id_logits = predictor.predict_logits(seq_info)
        if id_logits is None:
            return original_get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)

        id_logits = id_logits[0, 0, 0]
        if not runtime_tracker.use_sigmoid:
            id_scores = id_logits.softmax(dim=-1)
        else:
            id_scores = id_logits.sigmoid()
        if _has_assignment_ambiguity(id_scores, runtime_tracker, predictor.fallback_conf_margin):
            predictor.ambiguity_fallback_calls += 1
            return original_get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)

        match runtime_tracker.assignment_protocol:
            case "hungarian":
                id_labels = runtime_tracker._hungarian_assignment(id_scores=id_scores)
            case "object-max":
                id_labels = runtime_tracker._object_max_assignment(id_scores=id_scores)
            case "id-max":
                id_labels = runtime_tracker._id_max_assignment(id_scores=id_scores)
            case _:
                raise NotImplementedError
        return torch.tensor(id_labels, dtype=torch.int64, device=boxes.device)

    runtime_tracker._get_id_pred_labels = _get_id_pred_labels
    runtime_tracker.trt_id_predictor = predictor