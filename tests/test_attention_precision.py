import sys
from types import ModuleType
from unittest.mock import patch

import torch

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn_stub = ModuleType("flash_attn")
    flash_attn_stub.flash_attn_func = None
    flash_attn_stub.flash_attn_varlen_func = None
    with patch.dict(sys.modules, {"flash_attn": flash_attn_stub}):
        from weathergen.model import attention
else:
    from weathergen.model import attention


def test_varlen_attention_matches_flash_and_projection_dtypes(monkeypatch):
    def fake_flash_attention(query, key, value, *args, **kwargs):
        assert query.dtype == key.dtype == value.dtype == torch.bfloat16
        return value

    monkeypatch.setattr(attention, "flash_attn_varlen_func", fake_flash_attention)
    module = attention.MultiSelfAttentionHeadVarlen(
        dim_embed=8,
        num_heads=2,
        attention_dtype=torch.bfloat16,
    )

    result = module(torch.randn(4, 8, dtype=torch.float32), torch.tensor([4]))

    assert result.dtype == torch.float32
