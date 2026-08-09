# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from weathergen.model.norms import AdaLayerNorm, RMSNorm
from weathergen.model.positional_encoding import rotary_pos_emb_2d

"""
Attention blocks used by WeatherGenerator.

Some blocks optionally apply 2D RoPE. When enabled, the caller must provide per-token 2D
coordinates aligned with the token order (lat, lon in radians).
"""


class MultiSelfAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
        adaln_zero_init: bool = False,
        adaln_with_gate: bool = False,
    ):
        super(MultiSelfAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(
                dim_embed, dim_aux, norm_eps=norm_eps,
                zero_init=adaln_zero_init, with_gate=adaln_with_gate,
            )
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x, x_lens, ada_ln_aux=None, coords=None):
        if self.with_residual:
            x_in = x
        gate = None
        if ada_ln_aux is None:
            x = self.lnorm(x)
        else:
            out = self.lnorm(x, ada_ln_aux)
            x, gate = out if isinstance(out, tuple) else (out, None)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], self.num_heads, x.shape[-1] // self.num_heads]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        # x_lens arrives as [0, l0, l1, ...]: the leading 0 is the padding element
        # required by flash_attn's cu_seqlens API, NOT a real sequence. Only [1:]
        # are actual per-cell lengths. Filtering the full array with `> 0` silently
        # removes that pad and yields a cu_seqlens starting at l0 instead of 0, with
        # length batch instead of batch+1 -> flash reads garbage segment boundaries
        # -> NaN from finite inputs.
        cum_x_lens = torch.cumsum(x_lens, 0, dtype=torch.int32)

        lens = x_lens[1:]
        valid_mask = lens > 0

        if valid_mask.all():
            # Fast path: no empty sequences, pass cu_seqlens straight through.
            # This is the common case for dense grids and skips the Python loop below.
            outs = flash_attn_varlen_func(
                qs,
                ks,
                vs,
                cum_x_lens,
                cum_x_lens,
                int(lens.max()),
                int(lens.max()),
                softcap=self.softcap,
                dropout_p=dropout_rate,
            )
        elif valid_mask.any():
            valid_lens = lens[valid_mask]
            # rebuild cu_seqlens WITH the mandatory leading zero
            zero = torch.zeros(1, dtype=torch.int32, device=valid_lens.device)
            cum_valid_lens = torch.cat(
                [zero, torch.cumsum(valid_lens, 0, dtype=torch.int32)]
            )
            # With lens = x_lens[1:], sequence i spans [cum_x_lens[i], cum_x_lens[i+1]).
            # No i-1 special case: cum_x_lens[0] == 0 by construction of the pad.
            valid_idx = torch.nonzero(valid_mask).flatten()
            keep = torch.cat([
                torch.arange(cum_x_lens[i], cum_x_lens[i + 1], device=qs.device)
                for i in valid_idx
            ])
            # ordering of tensors (seq, heads, embed) (differs from torch's flash attention impl)
            valid_outs = flash_attn_varlen_func(
                qs[keep],
                ks[keep],
                vs[keep],
                cum_valid_lens,
                cum_valid_lens,
                int(valid_lens.max()),
                int(valid_lens.max()),
                softcap=self.softcap,
                dropout_p=dropout_rate,
            )
            outs = torch.zeros_like(qs)
            outs[keep] = valid_outs
        else:
            outs = torch.zeros_like(qs)

        out = self.proj_out(outs.flatten(-2, -1))

        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
#        outs = flash_attn_varlen_func(
#            qs,
#            ks,
#            vs,
#            cum_x_lens,
#            cum_x_lens,
#            x_lens.max(),
#            x_lens.max(),
#            softcap=self.softcap,
#            dropout_p=dropout_rate,
#        )
#
#        out = self.proj_out(outs.flatten(-2, -1))

        if gate is not None:
            out = gate * out
        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadVarlenFlex(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiSelfAttentionHeadVarlenFlex, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)
        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

        def att(qs, ks, vs, x_mask):
            def sparsity_mask(score, b, h, q_idx, kv_idx):
                return (q_idx // 16) == (kv_idx % 16)

            return flex_attention(qs, ks, vs, score_mod=sparsity_mask)

        self.compiled_flex_attention = torch.compile(att, dynamic=False)

    def forward(self, x, x_lens=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], 1, self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([1, 2, 0, 3])

        outs = self.compiled_flex_attention(qs, ks, vs).transpose(1, 2).squeeze()

        out = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadLocal(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        qkv_len,
        block_factor,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
    ):
        super(MultiSelfAttentionHeadLocal, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported."

        # define block mask
        def mask_block_local(batch, head, idx_q, idx_kv):
            return (idx_q // block_factor) == (idx_kv // block_factor)

        self.block_mask = create_block_mask(
            mask_block_local, B=None, H=None, Q_LEN=qkv_len, KV_LEN=qkv_len
        )
        # compile for efficiency
        self.flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, x, coords=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads
        s = [x.shape[0], x.shape[1], self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([0, 2, 1, 3])

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        outs = self.flex_attention(qs, ks, vs, block_mask=self.block_mask).transpose(1, 2)

        out = self.proj_out(self.dropout(outs.flatten(-2, -1)))
        if self.with_residual:
            out = x_in + out

        return out


class MultiCrossAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        adaln_zero_init: bool = False,
        adaln_with_gate: bool = False,
    ):
        super(MultiCrossAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(
                dim_embed_q,
                dim_aux,
                norm_eps=norm_eps,
                zero_init=adaln_zero_init,
                with_gate=adaln_with_gate,
            )
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        gate = None
        if ada_ln_aux is None:
            x_q = self.lnorm_in_q(x_q)
        else:
            out = self.lnorm_in_q(x_q, ada_ln_aux)
            # AdaLayerNorm returns (x, gate) when with_gate is enabled, else x
            x_q, gate = out if isinstance(out, tuple) else (out, None)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype)
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

#        if x_kv_lens is not None:
#            cum_x_q_lens = torch.cumsum(x_q_lens, 0, dtype=torch.int32)
#            cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)
#
#            print(f"qs shape: {qs.shape}, dtype: {qs.dtype}")  # [total_tokens, n_heads, d_head]
#            print(f"max_seqlen_q: {x_q_lens.max()}, max_seqlen_k: {x_kv_lens.max()}")
#            seqlens = x_q_lens
#            print(f"min seqlen: {seqlens.min()}, max: {seqlens.max()}")
#            print(f"dim_head_proj: {self.dim_head_proj}")
#
#            outs = flash_attn_varlen_func(
#                qs,
#                ks,
#                vs,
#                cum_x_q_lens,
#                cum_x_kv_lens,
#                x_q_lens.max(),
#                x_kv_lens.max(),
#                softcap=self.softcap,
#                dropout_p=dropout_rate,
#            )
#        else:
#            assert False
        if x_kv_lens is not None:
            # x_q_lens / x_kv_lens arrive as [0, l0, l1, ...]: the leading 0 is the
            # padding element required by flash_attn's cu_seqlens API, NOT a real
            # sequence. Only [1:] are actual per-sequence lengths. Filtering with
            # `> 0` on the full array silently removes that pad and produces a
            # cu_seqlens that starts at l0 instead of 0 -> flash reads garbage
            # segment boundaries -> NaN from finite inputs.
            cum_x_q_lens = torch.cumsum(x_q_lens, 0, dtype=torch.int32)
            cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)

            q_lens = x_q_lens[1:]
            kv_lens = x_kv_lens[1:]

            # A sequence can be attended only if BOTH q and kv are non-empty.
            # Sequences with q > 0 but kv == 0 keep their zero output here and are
            # carried by the residual below (attend to nothing -> contribute nothing).
            valid_mask = (q_lens > 0) & (kv_lens > 0)

            if valid_mask.all():
                # Fast path: nothing to filter, pass the cu_seqlens straight through.
                # This is the common case for dense grids (every cell populated) and
                # avoids the Python loop below entirely.
                outs = flash_attn_varlen_func(
                    qs,
                    ks,
                    vs,
                    cum_x_q_lens,
                    cum_x_kv_lens,
                    int(q_lens.max()),
                    int(kv_lens.max()),
                    softcap=self.softcap,
                    dropout_p=dropout_rate,
                )
            elif valid_mask.any():
                valid_q_lens = q_lens[valid_mask]
                valid_kv_lens = kv_lens[valid_mask]

                # rebuild cu_seqlens WITH the mandatory leading zero
                zero = torch.zeros(1, dtype=torch.int32, device=valid_q_lens.device)
                cum_valid_q_lens = torch.cat(
                    [zero, torch.cumsum(valid_q_lens, 0, dtype=torch.int32)]
                )
                cum_valid_kv_lens = torch.cat(
                    [zero, torch.cumsum(valid_kv_lens, 0, dtype=torch.int32)]
                )

                # Select tokens of valid sequences. With q_lens = x_q_lens[1:],
                # sequence i spans [cum_x_q_lens[i], cum_x_q_lens[i+1]) -- no i-1
                # special case is needed, because cum_x_q_lens[0] == 0 by construction.
                valid_idx = torch.nonzero(valid_mask).flatten()
                q_keep = torch.cat([
                    torch.arange(cum_x_q_lens[i], cum_x_q_lens[i + 1], device=qs.device)
                    for i in valid_idx
                ])
                kv_keep = torch.cat([
                    torch.arange(cum_x_kv_lens[i], cum_x_kv_lens[i + 1], device=ks.device)
                    for i in valid_idx
                ])

                valid_outs = flash_attn_varlen_func(
                    qs[q_keep],
                    ks[kv_keep],
                    vs[kv_keep],
                    cum_valid_q_lens,
                    cum_valid_kv_lens,
                    int(valid_q_lens.max()),
                    int(valid_kv_lens.max()),
                    softcap=self.softcap,
                    dropout_p=dropout_rate,
                )

                # Scatter results back into the full output tensor
                outs = torch.zeros_like(qs)
                outs[q_keep] = valid_outs
            else:
                outs = torch.zeros_like(qs)
        else:
            # No kv lengths given: nothing to attend over. Previously this fell
            # through to `outs.flatten(...)` with `outs` unbound -> UnboundLocalError.
            outs = torch.zeros_like(qs)

        outs = self.proj_out(outs.flatten(-2, -1))

        if gate is not None:
            # gated residual (DiT-style): x <- x + g * sublayer(x)
            outs = gate * outs
        if self.with_residual:
            outs = x_q_in + outs

        return outs


class MultiCrossAttentionHeadVarlenSlicedQ(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_slices_q,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHeadVarlenSlicedQ, self).__init__()

        self.num_slices_q = num_slices_q
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        assert num_heads % num_slices_q == 0
        num_heads_r = num_heads
        self.proj_heads_q = torch.nn.ModuleList()
        for _ in range(num_slices_q):
            self.proj_heads_q.append(
                torch.nn.Linear(dim_embed_q, num_heads_r * self.dim_head_proj, bias=False)
            )
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        x_q = self.lnorm_in_q(x_q) if ada_ln_aux is None else self.lnorm_in_q(x_q, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = [
            self.lnorm_q(head_proj(x_q_i).reshape(s)).to(self.dtype)
            for head_proj, x_q_i in zip(self.proj_heads_q, x_q.transpose(1, 0), strict=False)
        ]
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        # DEBUG
        _zq = (x_q_lens == 0).sum().item()
        _zkv = (x_kv_lens == 0).sum().item()
        if _zq > 0 or _zkv > 0:
            print(f"NANDBG flash lens: q_zero={_zq} kv_zero={_zkv}", flush=True)

        cum_x_q_lens = torch.cumsum(x_q_lens, 0, dtype=torch.int32)
        cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)
        outs = []
        for _i, qs_i in enumerate(qs):
            outs += [
                flash_attn_varlen_func(
                    qs_i,
                    ks,
                    vs,
                    cum_x_q_lens,
                    cum_x_kv_lens,
                    x_q_lens.max(),
                    x_kv_lens.max(),
                    softcap=self.softcap,
                    dropout_p=dropout_rate,
                )
            ]

        outs = self.proj_out(torch.stack(outs).transpose(1, 0).flatten(-2, -1))
        if self.with_residual:
            outs = x_q_in + outs.reshape(x_q_in.shape)

        return outs


class MultiSelfAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        softcap=0.0,
        norm_type="LayerNorm",
        qk_norm_type=None,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
    ):
        super(MultiSelfAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        if with_flash:
            self.att = torch.nn.functional.scaled_dot_product_attention
        else:
            self.att = self.attention
            self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, coords=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=2)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs = flash_attn_func(qs, ks, vs, softcap=self.softcap, dropout_p=dropout_rate)

        out = self.proj_out(outs.flatten(-2, -1))
        if self.with_residual:
            out = out + x_in

        return out


class MultiCrossAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_residual = with_residual
        self.with_flash = with_flash

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        assert dim_embed_q % num_heads == 0
        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        self.att = torch.nn.functional.scaled_dot_product_attention
        self.softmax = torch.nn.Softmax(dim=-1)

    #########################################
    def forward(self, x_q, x_kv):
        if self.with_residual:
            x_q_in = x_q
        x_q, x_kv = self.lnorm_in_q(x_q), self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], -1, self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype).transpose(-3, -2)
        s = [x_kv.shape[0], -1, self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype).transpose(-3, -2)
        vs = self.proj_heads_v(x_kv).reshape(s).transpose(-3, -2)

        # correct ordering of tensors with seq dimension second but last is critical
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            outs = self.att(qs, ks, vs).transpose(2, 1)

        outs = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            outs = x_q_in + outs

        return outs
