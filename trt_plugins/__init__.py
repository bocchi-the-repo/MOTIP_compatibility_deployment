"""TensorRT plugin registration helpers for MOTIP."""

from .ms_deform_attn_trt import register_ms_deform_attn_trt

__all__ = ["register_ms_deform_attn_trt"]