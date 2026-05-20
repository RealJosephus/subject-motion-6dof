from __future__ import annotations

import torch
import torch.nn as nn


class RelativeCausalSelfAttention(nn.Module):
    """Self-attention with causal/local masking and relative distance bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_len = int(max_len)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rel_bias = nn.Embedding(self.max_len, num_heads)

    def _relative_mask_and_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(seq_len, device=device)
        distance = idx.view(seq_len, 1) - idx.view(1, seq_len)
        allowed = (distance >= 0) & (distance < self.max_len)
        rel_idx = distance.clamp(min=0, max=self.max_len - 1)
        bias = self.rel_bias(rel_idx).permute(2, 0, 1).unsqueeze(0)
        return allowed, bias.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        allowed, bias = self._relative_mask_and_bias(seq_len, x.device, scores.dtype)
        scores = scores + bias
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, seq_len), -torch.inf)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(batch, 1, 1, seq_len),
                -torch.inf,
            )

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.proj_drop(self.proj(out))


class VisualTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_len: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RelativeCausalSelfAttention(dim, num_heads, max_len, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisualTemporalModule(nn.Module):
    """Causal temporal context over recent SAM-3D pose tokens only."""

    def __init__(
        self,
        input_dim: int = 1024,
        context_dim: int = 512,
        max_len: int = 240,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = int(max_len)
        if self.max_len < 1:
            raise ValueError("max_len must be >= 1")

        self.input = nn.Sequential(
            nn.LayerNorm(input_dim, eps=1e-6),
            nn.Linear(input_dim, context_dim),
        )
        self.blocks = nn.ModuleList(
            [
                VisualTransformerBlock(
                    dim=context_dim,
                    num_heads=num_heads,
                    max_len=self.max_len,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(context_dim, eps=1e-6)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pose_tokens.dim() != 3:
            raise ValueError("pose_tokens must have shape [B, T, D]")

        batch, seq_len, _ = pose_tokens.shape
        if seq_len < 1:
            raise ValueError("pose_tokens must contain at least one token")

        x = self.input(pose_tokens)

        padding_mask = None
        if lengths is not None:
            lengths = lengths.to(device=x.device).clamp(min=0, max=seq_len)
            padding_mask = (
                torch.arange(seq_len, device=x.device).view(1, seq_len)
                >= lengths.view(batch, 1)
            )

        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)
        return self.norm(x)


class ActionAutoregressiveHead(nn.Module):
    """Per-channel AR head whose recurrent state is updated from actions only."""

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_scale: float = 100.0,
        initial_value: float = 50.0,
        detach_feedback: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.output_scale = float(output_scale)
        self.initial_value = float(initial_value)
        self.detach_feedback = bool(detach_feedback)

        self.visual_input = nn.Sequential(
            nn.LayerNorm(context_dim, eps=1e-6),
            nn.Linear(context_dim, hidden_dim),
        )
        self.value_input = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.channel_embedding = nn.Embedding(output_dim, hidden_dim)
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def initial_actions(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.full(
            (batch, self.output_dim),
            self.initial_value,
            device=device,
            dtype=dtype,
        )

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [
            torch.zeros(
                batch,
                self.output_dim,
                self.hidden_dim,
                device=device,
                dtype=dtype,
            )
            for _ in self.cells
        ]

    def _feedback_embedding(
        self,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        feedback = prev_actions.detach() if self.detach_feedback else prev_actions
        feedback = feedback.clamp(0.0, self.output_scale)
        feedback = (feedback / self.output_scale * 2.0) - 1.0
        value = self.value_input(feedback.unsqueeze(-1))
        channel = self.channel_embedding.weight.to(dtype=prev_actions.dtype).unsqueeze(0)
        return value + channel

    def step(
        self,
        visual_context: torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if visual_context.dim() != 2:
            raise ValueError("visual_context must have shape [B, D]")

        batch = visual_context.shape[0]
        if prev_actions is None:
            prev_actions = self.initial_actions(
                batch, visual_context.device, visual_context.dtype
            )
        else:
            prev_actions = prev_actions.to(
                device=visual_context.device, dtype=visual_context.dtype
            )
        if action_state is None:
            action_state = self.initial_state(
                batch, visual_context.device, visual_context.dtype
            )

        action_input = self._feedback_embedding(prev_actions)
        visual = self.visual_input(visual_context).unsqueeze(1)

        pred_features = action_state[-1] + action_input + visual
        pred = self.head(self.norm(pred_features.reshape(batch * self.output_dim, -1)))
        pred = pred.view(batch, self.output_dim).sigmoid() * self.output_scale

        flat_x = action_input.reshape(batch * self.output_dim, self.hidden_dim)
        new_state = []
        for idx, cell in enumerate(self.cells):
            flat_h = action_state[idx].reshape(batch * self.output_dim, self.hidden_dim)
            flat_next = cell(flat_x, flat_h)
            next_h = flat_next.view(batch, self.output_dim, self.hidden_dim)
            if valid is not None:
                valid_view = valid.to(
                    device=visual_context.device, dtype=torch.bool
                ).view(batch, 1, 1)
                next_h = torch.where(valid_view, next_h, action_state[idx])
            new_state.append(next_h)
            flat_x = self.dropout(flat_next)

        if valid is not None:
            valid_view = valid.to(device=visual_context.device, dtype=torch.bool).view(
                batch, 1
            )
            pred = torch.where(valid_view, pred, torch.zeros_like(pred))
        return pred, new_state

    def forward(
        self,
        visual_context: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        bsz, seq_len, _ = visual_context.shape
        prev_actions = initial_actions
        if prev_actions is None:
            prev_actions = self.initial_actions(
                bsz, visual_context.device, visual_context.dtype
            )
        if action_state is None:
            action_state = self.initial_state(
                bsz, visual_context.device, visual_context.dtype
            )

        preds = []
        for idx in range(seq_len):
            valid = None
            if lengths is not None:
                valid = idx < lengths
            pred, action_state = self.step(
                visual_context[:, idx],
                prev_actions=prev_actions,
                action_state=action_state,
                valid=valid,
            )
            preds.append(pred)
            if valid is None:
                prev_actions = pred
            else:
                valid_view = valid.to(
                    device=visual_context.device, dtype=torch.bool
                ).view(bsz, 1)
                prev_actions = torch.where(valid_view, pred, prev_actions)

        pred_seq = torch.stack(preds, dim=1)
        if return_state:
            return pred_seq, prev_actions, action_state
        return pred_seq


class TokenBufferedTemporalHead(nn.Module):
    """Combines visual token context with action-only autoregressive state."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_scale: float = 100.0,
        initial_value: float = 50.0,
        detach_feedback: bool = False,
        max_len: int = 240,
        visual_num_layers: int = 2,
        visual_num_heads: int = 8,
        visual_ffn_dim: int = 2048,
    ):
        super().__init__()
        self.max_len = int(max_len)
        if self.max_len < 1:
            raise ValueError("max_len must be >= 1")

        self.visual_temporal = VisualTemporalModule(
            input_dim=input_dim,
            context_dim=hidden_dim,
            max_len=max_len,
            num_layers=visual_num_layers,
            num_heads=visual_num_heads,
            ffn_dim=visual_ffn_dim,
            dropout=dropout,
        )
        self.action_head = ActionAutoregressiveHead(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_scale=output_scale,
            initial_value=initial_value,
            detach_feedback=detach_feedback,
        )

    def initial_actions(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.action_head.initial_actions(batch, device, dtype)

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return self.action_head.initial_state(batch, device, dtype)

    def _trim_buffer(self, pose_token_buffer: torch.Tensor) -> torch.Tensor:
        if pose_token_buffer.shape[1] <= self.max_len:
            return pose_token_buffer
        return pose_token_buffer[:, -self.max_len :]

    def _append_pose_token(
        self,
        pose_token: torch.Tensor,
        pose_token_buffer: torch.Tensor | None,
    ) -> torch.Tensor:
        if pose_token.dim() != 2:
            raise ValueError("pose_token must have shape [B, D]")
        token = pose_token.unsqueeze(1)
        if pose_token_buffer is None:
            return token
        pose_token_buffer = pose_token_buffer.to(device=pose_token.device, dtype=pose_token.dtype)
        return self._trim_buffer(torch.cat([pose_token_buffer, token], dim=1))

    def _final_buffer(
        self,
        combined_tokens: torch.Tensor,
        prefix_len: int,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if lengths is None:
            return self._trim_buffer(combined_tokens)

        buffers = []
        lengths_cpu = lengths.detach().cpu().tolist()
        for batch_idx, valid_len in enumerate(lengths_cpu):
            valid_total = prefix_len + int(valid_len)
            tokens = combined_tokens[batch_idx, :valid_total]
            buffers.append(tokens[-self.max_len :])

        max_buffer_len = max(buffer.shape[0] for buffer in buffers)
        out = combined_tokens.new_zeros(
            combined_tokens.shape[0], max_buffer_len, combined_tokens.shape[-1]
        )
        for batch_idx, buffer in enumerate(buffers):
            out[batch_idx, -buffer.shape[0] :] = buffer
        return out

    def step(
        self,
        pose_token: torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        pose_token_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        pose_token_buffer = self._append_pose_token(pose_token, pose_token_buffer)
        visual_context = self.visual_temporal(pose_token_buffer)[:, -1]
        pred, action_state = self.action_head.step(
            visual_context,
            prev_actions=prev_actions,
            action_state=action_state,
        )
        return pred, action_state, pose_token_buffer

    def forward(
        self,
        pose_tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        pose_token_buffer: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]
    ):
        if pose_tokens.dim() != 3:
            raise ValueError("pose_tokens must have shape [B, T, D]")
        if lengths is not None:
            lengths = lengths.to(device=pose_tokens.device)

        prefix_len = 0
        if pose_token_buffer is not None:
            pose_token_buffer = self._trim_buffer(
                pose_token_buffer.to(device=pose_tokens.device, dtype=pose_tokens.dtype)
            )
            prefix_len = pose_token_buffer.shape[1]
            combined_tokens = torch.cat([pose_token_buffer, pose_tokens], dim=1)
        else:
            combined_tokens = pose_tokens

        context_lengths = None
        if lengths is not None:
            context_lengths = lengths + prefix_len

        visual_context = self.visual_temporal(
            combined_tokens,
            lengths=context_lengths,
        )[:, prefix_len : prefix_len + pose_tokens.shape[1]]
        pred, final_actions, final_action_state = self.action_head(
            visual_context,
            lengths=lengths,
            initial_actions=initial_actions,
            action_state=action_state,
            return_state=True,
        )

        if return_state:
            final_buffer = self._final_buffer(combined_tokens, prefix_len, lengths)
            return pred, final_actions, final_action_state, final_buffer
        return pred
