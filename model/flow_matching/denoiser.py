"""
HandFlow Rectified Flow Denoiser — all-frame direct regression.

Pose (48D axis-angle) and Trans (3D) are both regressed per-frame directly,
without velocity/cumsum. Beta (10D) serves as a global token.

Flow Target layout (flat packing per window, managed by mano_slice):
  x_1 = [beta(10) | pose_0(48) | ... | pose_{T-1}(48) | trans_0(3) | ... | trans_{T-1}(3)]
  flat_dim = 10 + T × 51

z-stream: 1 beta token + T pose tokens + T trans tokens = 1+2T tokens
c-stream: 2 × T tokens (1 image + 1 skeleton per frame, with cmask)

Condition pipeline:
  HaMeR backbone (frozen) → FrameCompressor (8-layer cross-attention pooling) → image tokens
  2D skeleton → ray direction → PE → MLP → skeleton tokens
  image + skeleton → cmask → condition stream
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from model.flow_matching.dualstream_transformer import (
    FlowMatchingTransformer,
    MLPEmbedder,
)
from model.flow_matching.helpers.model_wrapper import ModelWrapper
from model.flow_matching.helpers.scheduler import FluxTimeSampler
from model.feature_extractors.hamer_extractor import HaMeRBackbone
from model.feature_extractors.image_refiner import FrameCompressor
from model.feature_extractors.skeleton_encoder import RayDirectionSkeletonEncoder
from model.feature_extractors.condition_builder import ConditionBuilder
from model.pose_embedding import RotaryEmbedding


class ManoSlice(nn.Module):
    """Manage slice positions and normalization of each component in the flow target flat vector.

    Flat layout: [beta(10) | poses(T,48) | trans(T,3)]
    flat_dim = 10 + T * 51
    """

    def __init__(self, beta_dim: int, pose_dim: int,
                 trans_dim: int, window_size: int, stats_path: str | None = None):
        super().__init__()
        self.beta_dim = beta_dim       # 10
        self.pose_dim = pose_dim       # 48
        self.trans_dim = trans_dim     # 3
        self.window_size = window_size

        self.flat_dim = beta_dim + window_size * (pose_dim + trans_dim)

        # Build slices
        s = 0
        self.beta = slice(s, s + beta_dim)                                # [0:10]
        s += beta_dim
        self.poses = slice(s, s + window_size * pose_dim)                 # [10:10+48T]
        s += window_size * pose_dim
        self.trans = slice(s, s + window_size * trans_dim)                # [10+48T:10+51T]

        # Normalization statistics
        flat_mean = torch.zeros(self.flat_dim)
        flat_std = torch.ones(self.flat_dim)
        if stats_path is not None:
            import numpy as np
            stats = np.load(stats_path)
            beta_mean = torch.from_numpy(stats["beta_mean"])         # (10,)
            beta_std = torch.from_numpy(stats["beta_std"])           # (10,)
            pose_mean = torch.from_numpy(stats["pose_mean"])         # (48,)
            pose_std = torch.from_numpy(stats["pose_std"])           # (48,)
            trans_mean = torch.from_numpy(stats["trans_mean"])       # (3,)
            trans_std = torch.from_numpy(stats["trans_std"])         # (3,)
            flat_mean = torch.cat([
                beta_mean,
                pose_mean.repeat(window_size),
                trans_mean.repeat(window_size),
            ])
            flat_std = torch.cat([
                beta_std,
                pose_std.repeat(window_size),
                trans_std.repeat(window_size),
            ])
        self.register_buffer("flat_mean", flat_mean)
        self.register_buffer("flat_std", flat_std)

        # per-component statistics for auxiliary loss denormalization
        self.register_buffer("beta_mean_", flat_mean[self.beta])
        self.register_buffer("beta_std_", flat_std[self.beta])

        if stats_path is not None:
            self.register_buffer("pose_mean_", pose_mean)
            self.register_buffer("pose_std_", pose_std)
            self.register_buffer("trans_mean_", trans_mean)
            self.register_buffer("trans_std_", trans_std)
        else:
            self.register_buffer("pose_mean_", torch.zeros(pose_dim))
            self.register_buffer("pose_std_", torch.ones(pose_dim))
            self.register_buffer("trans_mean_", torch.zeros(trans_dim))
            self.register_buffer("trans_std_", torch.ones(trans_dim))

    def normalize(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Normalize the flat vector to ~N(0,1)."""
        return (x_flat - self.flat_mean) / self.flat_std

    def denormalize(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Denormalize the flat vector back to original scale."""
        return x_flat * self.flat_std + self.flat_mean

    def denormalize_components(
        self, beta: torch.Tensor, poses: torch.Tensor, trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Denormalize each component back to original scale."""
        beta_orig = beta * self.beta_std_ + self.beta_mean_
        poses_orig = poses * self.pose_std_ + self.pose_mean_    # (48,) broadcast
        trans_orig = trans * self.trans_std_ + self.trans_mean_  # (3,) broadcast
        return beta_orig, poses_orig, trans_orig

    def unpack(self, x_flat: torch.Tensor):
        """Unpack flat → (beta, poses, trans)"""
        B = x_flat.shape[0]
        beta = x_flat[:, self.beta]       # (B, 10)
        poses = x_flat[:, self.poses].reshape(B, self.window_size, self.pose_dim)    # (B, T, 48)
        trans = x_flat[:, self.trans].reshape(B, self.window_size, self.trans_dim)   # (B, T, 3)
        return beta, poses, trans

    def pack(self, beta: torch.Tensor, poses: torch.Tensor,
             trans: torch.Tensor) -> torch.Tensor:
        """Pack → flat"""
        B = beta.shape[0]
        return torch.cat([
            beta.reshape(B, -1),
            poses.reshape(B, -1),
            trans.reshape(B, -1),
        ], dim=-1)


class HandPoseDenoiser(nn.Module):
    """
    Rectified Flow Denoiser — all-frame direct regression.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        fm_cfg = config.model.fm
        cond_cfg = fm_cfg.condition
        comp_cfg = fm_cfg.frame_compressor
        tf_cfg = fm_cfg.transformer

        self.d_model = tf_cfg.hidden_size
        self.beta_dim = fm_cfg.beta_dim       # 10
        self.pose_dim = fm_cfg.pose_dim       # 48
        self.trans_dim = fm_cfg.trans_dim     # 3
        self.window_size = config.dataset.window_size

        # mano_slice (includes normalization statistics)
        stats_path = fm_cfg.get("normalization_stats", None)
        self.mano_slice = ManoSlice(
            beta_dim=self.beta_dim,
            pose_dim=self.pose_dim,
            trans_dim=self.trans_dim,
            window_size=self.window_size,
            stats_path=stats_path,
        )

        # z-stream input projections: each component → d_model
        self.beta_proj_in = nn.Linear(self.beta_dim, self.d_model)
        self.pose_proj_in = nn.Linear(self.pose_dim, self.d_model)
        self.trans_proj_in = nn.Linear(self.trans_dim, self.d_model)

        # z-stream output projections: d_model → native dims
        self.beta_proj_out = nn.Linear(self.d_model, self.beta_dim)
        self.pose_proj_out = nn.Linear(self.d_model, self.pose_dim)
        self.trans_proj_out = nn.Linear(self.d_model, self.trans_dim)

        # HaMeR backbone (frozen)
        hamer_ckpt = config.model.get("hamer", {}).get("checkpoint", None)
        self.hamer_backbone = HaMeRBackbone(checkpoint_path=hamer_ckpt)

        # Frame Compressor (directly compresses HaMeR patch tokens)
        self.frame_compressor = FrameCompressor(
            d_model=self.d_model,
            input_dim=cond_cfg.hamer_embed_dim,
            num_heads=tf_cfg.num_heads,
            num_layers=comp_cfg.num_layers,
        )

        # Skeleton Encoder
        self.skeleton_encoder = RayDirectionSkeletonEncoder(
            n_joints=cond_cfg.n_joints,
            num_freqs=cond_cfg.skeleton_pe_freqs,
            embed_dim=self.d_model,
        )

        # Condition Builder (cmask; mask_ratio only used for optional regularization)
        self.condition_builder = ConditionBuilder(
            d_model=self.d_model,
            mask_ratio=cond_cfg.get("cmask_ratio", 0.0),
        )

        # RoPE for Denoiser
        self.rope = RotaryEmbedding(dim=self.d_model // tf_cfg.num_heads)

        # Flow Matching Transformer
        self.transformer = FlowMatchingTransformer(
            in_channels=self.d_model,
            out_channels=self.d_model,
            use_context_in=True,
            context_in_dim=self.d_model,
            use_txt_in=False,
            vec_in_dim=0,
            use_pre_text_attn=False,
            config=tf_cfg,
        )

    def _extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Image feature extraction: HaMeR backbone → Frame Compressor.
        images: (B, T, 3, 256, 256) uint8
        Returns: (B, T, d_model) — 1 image token per frame
        """
        B, T = images.shape[:2]
        imgs = images.reshape(B * T, 3, images.shape[-2], images.shape[-1]).float() / 255.0
        with torch.no_grad():
            patches = self.hamer_backbone(imgs)
        patches = patches.reshape(B, T, 192, -1)
        compressed = self.frame_compressor(patches)
        return compressed

    def _build_condition(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the condition stream and return (condition, m)."""
        if "image_tokens" in batch:
            image_tokens = batch["image_tokens"]
        else:
            image_tokens = self._extract_image_features(batch["images"])
        skeleton_tokens = self.skeleton_encoder(
            batch["hamer_landmarks"],
            batch["crop_intrinsics"],
        )
        condition, m = self.condition_builder(
            image_tokens,
            skeleton_tokens,
            batch["hamer_confidence"],
        )
        return condition, m

    def _build_z_positions(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """z-stream RoPE positions: [0, 1, 1, 2, 2, ..., T, T]"""
        pos = torch.zeros(1 + 2 * T, dtype=torch.long, device=device)
        for t in range(T):
            pos[1 + t] = t + 1           # pose token
            pos[1 + T + t] = t + 1       # trans token
        return pos

    def _build_cond_positions(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """c-stream RoPE positions: the 2 tokens per frame share position t"""
        pos = torch.zeros(2 * T, dtype=torch.long, device=device)
        for t in range(T):
            pos[2 * t] = t
            pos[2 * t + 1] = t
        return pos

    def forward(self, x_t_flat: torch.Tensor, t: torch.Tensor, batch: dict):
        """
        Args:
            x_t_flat: (B, flat_dim) — noisy flow target
            t: (B,) — flow timestep
            batch: dict with all condition data
        Returns:
            v_pred_flat: (B, flat_dim) — predicted flow velocity
        """
        B, T = x_t_flat.shape[0], self.window_size
        device = x_t_flat.device

        # 1. Unpack flat → components
        beta_t, poses_t, trans_t = self.mano_slice.unpack(x_t_flat)

        # 2. Project to d_model → z-stream tokens
        beta_tok = self.beta_proj_in(beta_t).unsqueeze(1)           # (B, 1, D)
        pose_tok = self.pose_proj_in(poses_t)                       # (B, T, D)
        trans_tok = self.trans_proj_in(trans_t)                     # (B, T, D)
        z_stream = torch.cat([beta_tok, pose_tok, trans_tok], dim=1)  # (B, 1+2T, D)

        # 3. Build condition → c-stream
        condition, _ = self._build_condition(batch)  # (B, 2*T, D)

        # 4. RoPE positions
        z_positions = self._build_z_positions(B, T, device)
        cond_positions = self._build_cond_positions(B, T, device)

        # 5. Transformer forward
        pc_out, _ = self.transformer(
            pc=z_stream,
            dino=condition,
            timesteps=t,
            y=None,
            txt_tokens=None,
            z_positions=z_positions,
            cond_positions=cond_positions,
            rope_module=self.rope,
        )

        # 6. Project back to native dims
        v_beta = self.beta_proj_out(pc_out[:, 0:1])          # (B, 1, 10)
        v_pose = self.pose_proj_out(pc_out[:, 1:1 + T])      # (B, T, 48)
        v_trans = self.trans_proj_out(pc_out[:, 1 + T:])     # (B, T, 3)

        # 7. Pack flat
        v_pred_flat = self.mano_slice.pack(
            v_beta.squeeze(1), v_pose, v_trans
        )
        return v_pred_flat

    @torch.no_grad()
    def build_flow_target(self, batch: dict) -> torch.Tensor:
        """Build the GT flow target x_1 (normalized space) from the batch."""
        beta = batch["mano_betas"]       # (B, 10)
        poses = batch["mano_params"]     # (B, T, 48)
        trans = batch["mano_trans"]      # (B, T, 3)
        flat = self.mano_slice.pack(beta, poses, trans)
        return self.mano_slice.normalize(flat)

    @torch.no_grad()
    def reconstruct_params(self, x_1_flat: torch.Tensor):
        """
        Reconstruct MANO parameters (original scale) from the predicted x_1 (normalized space).

        All values are used directly, without cumsum.

        Returns:
            params: (B, T, 51) = cat[pose(48), trans(3)]
            beta: (B, 10)
        """
        x_1_orig = self.mano_slice.denormalize(x_1_flat)
        beta, poses, trans = self.mano_slice.unpack(x_1_orig)
        params = torch.cat([poses, trans], dim=-1)   # (B, T, 51)
        return params, beta

    def get_model_wrapper(self):
        """Return a ModelWrapper for the ODE solver."""
        return ModelWrapper(self)

    def get_time_sampler(self):
        """Return the time sampler."""
        return FluxTimeSampler()
