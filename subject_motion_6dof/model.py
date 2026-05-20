from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from sam_3d_body.models.backbones import create_backbone
from sam_3d_body.models.decoders.prompt_encoder import PromptEncoder
from sam_3d_body.models.decoders.promptable_decoder import PromptableDecoder
from sam_3d_body.models.modules.camera_embed import CameraEncoder

from .config import CfgNode
from .temporal import TokenBufferedTemporalHead


class SubjectMotion6DoFModel(nn.Module):
    """SAM-3D pose tokens plus token-buffered visual/action temporal heads."""

    pose_dim = 519
    cam_dim = 3
    cond_dim = 3

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.image_size = tuple(cfg.MODEL.IMAGE_SIZE)
        self.register_buffer(
            "image_mean",
            torch.tensor(cfg.MODEL.IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(cfg.MODEL.IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.backbone = create_backbone(cfg.MODEL.BACKBONE.TYPE, cfg)
        dec_cfg = cfg.MODEL.DECODER
        self.decoder = PromptableDecoder(
            dims=dec_cfg.DIM,
            context_dims=self.backbone.embed_dims,
            depth=dec_cfg.DEPTH,
            num_heads=dec_cfg.HEADS,
            head_dims=dec_cfg.DIM_HEAD,
            mlp_dims=dec_cfg.MLP_DIM,
            layer_scale_init_value=dec_cfg.LAYER_SCALE_INIT,
            drop_rate=dec_cfg.DROP_RATE,
            attn_drop_rate=dec_cfg.ATTN_DROP_RATE,
            drop_path_rate=dec_cfg.DROP_PATH_RATE,
            ffn_type=dec_cfg.FFN_TYPE,
            enable_twoway=dec_cfg.ENABLE_TWOWAY,
            repeat_pe=dec_cfg.REPEAT_PE,
            frozen=False,
            do_interm_preds=False,
            do_keypoint_tokens=dec_cfg.get("DO_KEYPOINT_TOKENS", True),
            keypoint_token_update=False,
        )

        self.init_pose = nn.Embedding(1, self.pose_dim)
        self.init_camera = nn.Embedding(1, self.cam_dim)
        nn.init.zeros_(self.init_camera.weight)
        init_dim = self.pose_dim + self.cam_dim + self.cond_dim
        self.init_to_token_mhr = nn.Linear(init_dim, dec_cfg.DIM)
        self.prev_to_token_mhr = nn.Linear(init_dim - self.cond_dim, dec_cfg.DIM)

        self.prompt_encoder = PromptEncoder(
            embed_dim=self.backbone.embed_dims,
            num_body_joints=70,
            frozen=False,
            mask_embed_type=cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None),
        )
        self.prompt_to_token = nn.Linear(self.backbone.embed_dims, dec_cfg.DIM)
        self.ray_cond_emb = CameraEncoder(self.backbone.embed_dim, self.backbone.patch_size)

        self.keypoint_embedding = nn.Embedding(70, dec_cfg.DIM)
        self.keypoint3d_embedding = nn.Embedding(70, dec_cfg.DIM)

        head_cfg = cfg.TEMPORAL_HEAD
        self.temporal_head = TokenBufferedTemporalHead(
            input_dim=head_cfg.INPUT_DIM,
            hidden_dim=head_cfg.HIDDEN_DIM,
            output_dim=head_cfg.OUTPUT_DIM,
            num_layers=head_cfg.NUM_LAYERS,
            dropout=head_cfg.DROPOUT,
            output_scale=head_cfg.OUTPUT_SCALE,
            initial_value=float(head_cfg.INITIAL_VALUE),
            detach_feedback=bool(head_cfg.DETACH_FEEDBACK),
            max_len=int(cfg.DATA.MAX_LEN),
            visual_num_layers=int(head_cfg.get("VISUAL_NUM_LAYERS", 2)),
            visual_num_heads=int(head_cfg.get("VISUAL_NUM_HEADS", 8)),
            visual_ffn_dim=int(head_cfg.get("VISUAL_FFN_DIM", head_cfg.HIDDEN_DIM * 4)),
        )

        self.freeze_backbone = bool(cfg.TRAINING.get("FREEZE_BACKBONE", True))
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        encoder = getattr(self.backbone, "encoder", None)
        if hasattr(encoder, "set_grad_checkpointing"):
            encoder.set_grad_checkpointing(enabled)
        elif hasattr(encoder, "gradient_checkpointing"):
            encoder.gradient_checkpointing = enabled

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.max() > 2.0:
            frames = frames / 255.0
        return (frames - self.image_mean.to(frames)) / self.image_std.to(frames)

    def _ray_condition(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        focal = math.sqrt(float(height * height + width * width))
        rays = torch.stack([(xs - width * 0.5) / focal, (ys - height * 0.5) / focal], dim=0)
        return rays.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    def _condition_info(self, batch: int, device, dtype) -> torch.Tensor:
        # CLIFF-style full-frame condition: centered crop, full-frame size / focal.
        h, w = self.image_size[1], self.image_size[0]
        scale = max(h, w) / math.sqrt(float(h * h + w * w))
        cond = torch.tensor([0.0, 0.0, scale], device=device, dtype=dtype)
        return cond.view(1, 3).expand(batch, -1).contiguous()

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Return per-frame SAM-3D primary body token, shape [B, T, 1024]."""
        if frames.dim() != 5:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        bsz, seq_len, channels, height, width = frames.shape
        flat = frames.view(bsz * seq_len, channels, height, width)
        flat = self._preprocess(flat)

        backbone_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with backbone_ctx:
            image_embeddings = self.backbone(flat)
            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.to(dtype=flat.dtype)

        ray_cond = self._ray_condition(
            bsz * seq_len, height, width, flat.device, flat.dtype
        )
        image_embeddings = self.ray_cond_emb(image_embeddings, ray_cond)

        prompt_pe = self.prompt_encoder.get_dense_pe(image_embeddings.shape[-2:]).to(
            image_embeddings
        )
        keypoints = torch.zeros((bsz * seq_len, 1, 3), device=flat.device, dtype=flat.dtype)
        keypoints[..., -1] = -2
        prompt_embeddings, _ = self.prompt_encoder(keypoints=keypoints)
        prompt_embeddings = self.prompt_to_token(prompt_embeddings)

        init_pose = self.init_pose.weight.expand(bsz * seq_len, -1).unsqueeze(1)
        init_camera = self.init_camera.weight.expand(bsz * seq_len, -1).unsqueeze(1)
        init_estimate = torch.cat([init_pose, init_camera], dim=-1)
        condition_info = self._condition_info(bsz * seq_len, flat.device, flat.dtype)
        init_input = torch.cat([condition_info[:, None, :], init_estimate], dim=-1)
        pose_token = self.init_to_token_mhr(init_input)
        prev_token = self.prev_to_token_mhr(init_estimate)

        token_embeddings = torch.cat(
            [
                pose_token,
                prev_token,
                prompt_embeddings,
                self.keypoint_embedding.weight[None].expand(bsz * seq_len, -1, -1),
                self.keypoint3d_embedding.weight[None].expand(bsz * seq_len, -1, -1),
            ],
            dim=1,
        )
        token_augment = torch.zeros_like(token_embeddings)
        token_augment[:, 1:2] = prev_token
        token_augment[:, 2:3] = prompt_embeddings

        tokens = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment=token_augment,
            image_augment=prompt_pe,
            token_mask=None,
        )
        pose_tokens = tokens[:, 0].reshape(bsz, seq_len, -1)
        return pose_tokens

    def predict_actions_step(
        self,
        pose_token: torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        pose_token_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        return self.temporal_head.step(
            pose_token,
            prev_actions=prev_actions,
            action_state=action_state,
            pose_token_buffer=pose_token_buffer,
        )

    def forward(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_actions: torch.Tensor | None = None,
        action_state: list[torch.Tensor] | None = None,
        pose_token_buffer: torch.Tensor | None = None,
    ) -> dict:
        pose_tokens = self.encode_frames(frames)
        pred, final_actions, final_action_state, final_pose_token_buffer = self.temporal_head(
            pose_tokens,
            lengths=lengths,
            initial_actions=initial_actions,
            action_state=action_state,
            pose_token_buffer=pose_token_buffer,
            return_state=True,
        )
        return {
            "pred_actions": pred,
            "pose_tokens": pose_tokens,
            "prev_actions": final_actions,
            "action_state": final_action_state,
            "pose_token_buffer": final_pose_token_buffer,
        }


def build_model(cfg: CfgNode) -> SubjectMotion6DoFModel:
    model = SubjectMotion6DoFModel(cfg)
    if cfg.TRAINING.get("GRADIENT_CHECKPOINTING", True):
        model.set_gradient_checkpointing(True)
    return model
