"""Opt-in TensorRT wrapper and builder for MOTIP DETR inference."""

import types
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _get_trt_logger(trt):
    logger = getattr(trt, "_motip_logger", None)
    if logger is None:
        logger = trt.Logger(trt.Logger.ERROR)
        setattr(trt, "_motip_logger", logger)
    return logger


class _DetrExportWrapper(nn.Module):
    def __init__(self, detr: nn.Module):
        super().__init__()
        self.detr = detr

    def forward(self, tensors: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from motip.utils.nested_tensor import NestedTensor

        outputs = self.detr(NestedTensor(tensors=tensors, mask=mask))
        return outputs["pred_logits"], outputs["pred_boxes"], outputs["outputs"]


def _install_msda_symbolic() -> None:
    from motip.models.ops.functions import MSDeformAttnFunction

    def output_type(value: Any, sampling_locations: Any) -> Any:
        value_type = value.type()
        value_sizes = value_type.sizes()
        sampling_sizes = sampling_locations.type().sizes()
        if value_sizes is None or sampling_sizes is None:
            return value_type
        if len(value_sizes) < 4 or len(sampling_sizes) < 2:
            return value_type
        if value_sizes[2] is None or value_sizes[3] is None:
            return value_type
        output_sizes = [value_sizes[0], sampling_sizes[1], value_sizes[2] * value_sizes[3]]
        return value_type.with_sizes(output_sizes)

    def symbolic(g: Any, value: Any, value_spatial_shapes: Any, value_level_start_index: Any,
                 sampling_locations: Any, attention_weights: Any, im2col_step: Any) -> Any:
        output = g.op(
            "motip::MSDeformAttnTRT",
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step_i=64,
        )
        output.setType(output_type(value, sampling_locations))
        return output

    MSDeformAttnFunction.symbolic = staticmethod(symbolic)


def _install_static_export_adapter(detr: nn.Module, height: int, width: int,
                                   dtype: torch.dtype, device: torch.device) -> None:
    import torch.nn.functional as F

    from motip.models.ops.functions import MSDeformAttnFunction
    from motip.models.ops.modules import MSDeformAttn
    from motip.utils.nested_tensor import NestedTensor

    dummy_tensors = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
    dummy_mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
    with torch.no_grad():
        features, pos = detr.backbone(NestedTensor(tensors=dummy_tensors, mask=dummy_mask))
        srcs = []
        masks = []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(detr.input_proj[level](src))
            masks.append(mask)
        if detr.num_feature_levels > len(srcs):
            original_len = len(srcs)
            for level in range(original_len, detr.num_feature_levels):
                if level == original_len:
                    src = detr.input_proj[level](features[-1].tensors)
                else:
                    src = detr.input_proj[level](srcs[-1])
                mask = F.interpolate(dummy_mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                srcs.append(src)
                masks.append(mask)
                pos.append(detr.backbone[1](NestedTensor(src, mask)).to(src.dtype))

        spatial_shapes = torch.as_tensor(
            [(src.shape[-2], src.shape[-1]) for src in srcs], dtype=torch.long, device=device
        )
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones((1, len(srcs), 2), dtype=torch.float32, device=device)
        reference_points = detr.transformer.encoder.get_reference_points(spatial_shapes, valid_ratios, device=device)

    transformer = detr.transformer
    encoder = transformer.encoder
    transformer.register_buffer("_export_spatial_shapes", spatial_shapes, persistent=False)
    transformer.register_buffer("_export_level_start_index", level_start_index, persistent=False)
    transformer.register_buffer("_export_valid_ratios", valid_ratios, persistent=False)
    encoder.register_buffer("_export_reference_points", reference_points, persistent=False)

    def static_encoder_forward(self: Any, src: torch.Tensor, spatial_shapes_arg: torch.Tensor,
                               level_start_index_arg: torch.Tensor, valid_ratios_arg: torch.Tensor,
                               pos_arg: torch.Tensor | None = None,
                               padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = src
        reference_points_const = self._export_reference_points.to(device=src.device)
        spatial_shapes_const = transformer._export_spatial_shapes.to(device=src.device)
        level_start_index_const = transformer._export_level_start_index.to(device=src.device)
        for layer in self.layers:
            output = layer(output, pos_arg, reference_points_const, spatial_shapes_const,
                           level_start_index_const, padding_mask)
        return output

    def static_transformer_forward(self: Any, srcs: list[torch.Tensor], masks: list[torch.Tensor],
                                   pos_embeds: list[torch.Tensor],
                                   query_embed: torch.Tensor | None = None) -> tuple[Any, Any, Any, Any, Any]:
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        spatial_shapes_const = self._export_spatial_shapes.to(device=src_flatten.device)
        level_start_index_const = self._export_level_start_index.to(device=src_flatten.device)
        valid_ratios_const = self._export_valid_ratios.to(device=src_flatten.device)
        memory = self.encoder(src_flatten, spatial_shapes_const, level_start_index_const,
                              valid_ratios_const, lvl_pos_embed_flatten, mask_flatten)

        batch_size, _, channels = memory.shape
        if self.two_stage:
            raise RuntimeError("MOTIP TensorRT DETR export only supports the current one-stage config")
        query_embed_split, tgt = torch.split(query_embed, channels, dim=1)
        query_embed_split = query_embed_split.unsqueeze(0).expand(batch_size, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points(query_embed_split).sigmoid()
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes_const,
            level_start_index_const,
            valid_ratios_const,
            query_embed_split,
            mask_flatten,
        )
        return hs, reference_points, inter_references, None, None

    def export_ms_deform_attn_forward(self: Any, query: torch.Tensor, reference_points: torch.Tensor,
                                      input_flatten: torch.Tensor, input_spatial_shapes: torch.Tensor,
                                      input_level_start_index: torch.Tensor,
                                      input_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        _, input_len, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(batch_size, input_len, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, query_len, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, query_len, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, query_len, self.n_heads, self.n_levels, self.n_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise RuntimeError(f"Unsupported reference_points last dimension: {reference_points.shape[-1]}")

        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        return self.output_proj(output)

    encoder.forward = types.MethodType(static_encoder_forward, encoder)
    transformer.forward = types.MethodType(static_transformer_forward, transformer)
    for module in detr.modules():
        if isinstance(module, MSDeformAttn):
            module.forward = types.MethodType(export_ms_deform_attn_forward, module)


def _build_engine_from_onnx(onnx_path: Path, engine_path: Path, dtype: torch.dtype) -> None:
    import tensorrt as trt

    from motip.trt_plugins import register_ms_deform_attn_trt

    register_ms_deform_attn_trt()
    builder = trt.Builder(_get_trt_logger(trt))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, _get_trt_logger(trt))
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse MOTIP DETR ONNX for TensorRT:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build MOTIP DETR TensorRT engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def ensure_trt_detr_engine(detr: nn.Module, engine_path: Path, device: torch.device,
                           input_hw: tuple[int, int] = (640, 640),
                           dtype: torch.dtype = torch.float32) -> Path:
    engine_path = Path(engine_path).expanduser().resolve()
    if engine_path.exists():
        return engine_path

    height, width = input_hw
    onnx_path = engine_path.with_suffix(".onnx")
    tensors = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
    mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
    wrapper = _DetrExportWrapper(detr).eval()
    _install_static_export_adapter(detr, height, width, dtype, device)
    _install_msda_symbolic()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            (tensors, mask),
            str(onnx_path),
            input_names=["tensors", "mask"],
            output_names=["pred_logits", "pred_boxes", "outputs"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    _build_engine_from_onnx(onnx_path, engine_path, dtype)
    return engine_path


class TensorRTDetrWrapper(nn.Module):
    def __init__(self, engine_path: Path, device: torch.device):
        super().__init__()
        import tensorrt as trt

        from motip.trt_plugins import register_ms_deform_attn_trt

        register_ms_deform_attn_trt()
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.device = device
        self.trt = trt
        self.trt_calls = 0
        runtime = trt.Runtime(_get_trt_logger(trt))
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT DETR engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT DETR execution context")
        self.stream = torch.cuda.Stream(device=device)
        self.input_specs = self._read_input_specs()

    def _read_input_specs(self) -> dict[str, tuple[tuple[int, ...], Any]]:
        specs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                specs[name] = (tuple(self.engine.get_tensor_shape(name)), self.engine.get_tensor_dtype(name))
        if set(specs) != {"tensors", "mask"}:
            raise RuntimeError(f"Unexpected TensorRT DETR inputs: {sorted(specs)}")
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

    def _validate_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        expected_shape, expected_dtype = self.input_specs[name]
        expected_torch_dtype = self._torch_dtype_for_trt(expected_dtype)
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"TensorRT DETR input {name} expects shape {expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != expected_torch_dtype:
            raise RuntimeError(f"TensorRT DETR input {name} expects dtype {expected_torch_dtype}, got {tensor.dtype}")
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor.contiguous()

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
    def forward(self, samples: Any) -> dict[str, torch.Tensor]:
        tensors, mask = samples.decompose()
        bindings = {
            "tensors": self._validate_input("tensors", tensors),
            "mask": self._validate_input("mask", mask),
        }
        outputs = self._make_outputs()
        bindings.update(outputs)
        for name, tensor in bindings.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Failed to bind TensorRT DETR tensor address: {name}")

        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT DETR execute_async_v3 returned False")
        for tensor in outputs.values():
            tensor.record_stream(self.stream)
        current_stream.wait_stream(self.stream)
        result = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
            "outputs": outputs["outputs"],
        }
        self.trt_calls += 1
        return result

    def summary(self) -> str:
        return f"TensorRT DETR calls: {self.trt_calls}"