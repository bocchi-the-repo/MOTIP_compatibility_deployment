"""MOTIP-owned TensorRT plugin for multi-scale deformable attention."""

PLUGIN_ID = "motip::MSDeformAttnTRT"


def register_ms_deform_attn_trt() -> None:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    registry = trt.get_plugin_registry()
    if registry.get_creator("MSDeformAttnTRT", "1", "motip") is not None:
        return

    @trtp.register(PLUGIN_ID)
    def ms_deform_attn_trt_desc(
        value: trtp.TensorDesc,
        value_spatial_shapes: trtp.TensorDesc,
        value_level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        output_shape = (
            value.shape_expr[0],
            sampling_locations.shape_expr[1],
            value.shape_expr[2] * value.shape_expr[3],
        )
        return trtp.from_shape_expr(output_shape, value.dtype)

    @trtp.impl(PLUGIN_ID)
    def ms_deform_attn_trt_impl(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        outputs,
        stream,
    ) -> None:
        import os

        import torch
        import MultiScaleDeformableAttention as MSDA

        output = torch.as_tensor(outputs[0], device="cuda")
        value_tensor = torch.as_tensor(value, device="cuda")
        spatial_shapes = torch.as_tensor(value_spatial_shapes, device="cuda")
        level_start_index = torch.as_tensor(value_level_start_index, device="cuda")
        sampling_locations_tensor = torch.as_tensor(sampling_locations, device="cuda")
        attention_weights_tensor = torch.as_tensor(attention_weights, device="cuda")

        external_stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(external_stream):
            if spatial_shapes.dtype != torch.int64:
                spatial_shapes = spatial_shapes.to(torch.int64)
            if level_start_index.dtype != torch.int64:
                level_start_index = level_start_index.to(torch.int64)
            result = MSDA.ms_deform_attn_forward(
                value_tensor.contiguous(),
                spatial_shapes.contiguous(),
                level_start_index.contiguous(),
                sampling_locations_tensor.contiguous(),
                attention_weights_tensor.contiguous(),
                64,
            )
            result.record_stream(external_stream)
            output.copy_(result)
        if os.environ.get("MOTIP_TRT_PLUGIN_SYNC"):
            external_stream.synchronize()
