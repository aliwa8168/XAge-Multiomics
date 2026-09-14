from typing import Literal

import torch
import torch.nn as nn

from scflowdiff.core.layers import (
    Block,
    CrossAttentionBlock,
    FinalLayerDit,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
)

#############################
#        Like in scVI       #
#############################


class EncoderScvi(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_genes if i == 0 else n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder_mlp = nn.Sequential(*layers)
        self.latent_embedding = 0

    def forward(self, x: torch.Tensor, genes: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        x = torch.log1p(x)
        x = self.encoder_mlp(x)
        return x, None


class DecoderScvi(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_latent if i == 0 else n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.decoder_mlp = nn.Sequential(*layers)
        self.last_embedding = None

    def forward(self, x: torch.Tensor, condition: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.decoder_mlp(x)
        return x


#################################
#         All Transformer       #
#################################


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_inducing_points: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        positional_encoding: bool = False,
    ):
        super().__init__()

        self.latent_embedding = n_embed_latent
        self.latent_dim = n_inducing_points
        self.encoder_layers = nn.ModuleList([])
        self.pos_embed: torch.Tensor | None
        if positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_inducing_points, n_embed), requires_grad=False)
        else:
            self.pos_embed = None

        self.ca_layer = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=n_inducing_points,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
        )

        for _ in range(n_layer):
            self.encoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                )
            )

        self.encoder_latent_input = nn.Sequential(
            nn.Linear(n_embed, n_embed_latent, bias=bias),
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ca_layer(x, key_padding_mask=key_padding_mask)
        if isinstance(self.pos_embed, nn.Parameter):
            x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        h = self.encoder_latent_input(x)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        n_layer: int,
        n_inducing_points: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        shared_embedding: bool,
        use_adaln: bool = False,
    ):
        super().__init__()

        self.gene_embedding = nn.Embedding(n_genes + 1, n_embed) if not shared_embedding else nn.Identity()
        self.decoder_layers = nn.ModuleList([])

        self.decoder_latent_input = nn.Sequential(
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
            nn.Linear(n_embed_latent, n_embed, bias=bias),
        )

        for _ in range(n_layer):
            self.decoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=use_adaln,
                )
            )
        self.decoder_cross_attention = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=0,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
            use_adaln=use_adaln,
        )

    def forward(
        self, x: torch.Tensor, genes: torch.Tensor, condition: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        x = self.decoder_latent_input(x)
        for layer in self.decoder_layers:
            x = layer(x, condition)
        q = self.gene_embedding(genes)
        output = self.decoder_cross_attention(x, q, condition)
        return output


####################
#        DiT       #
####################


class DiT(nn.Module):
    """Diffusion Transformer."""

    def __init__(
        self,
        n_embed: int,
        n_embed_input: int,
        n_layer: int,
        n_head: int,
        seq_len: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        class_vocab_sizes: dict[str, int],
        cfg_dropout_prob: float = 0.1,
        condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive",
    ):
        super().__init__()
        self.class_vocab_sizes = class_vocab_sizes
        self.cfg_dropout_prob = cfg_dropout_prob
        self.condition_strategy = condition_strategy
        self.class_embeddings = nn.ModuleDict()
        for class_name, vocab_size in class_vocab_sizes.items():
            use_cfg_embedding = int(cfg_dropout_prob > 0)
            self.class_embeddings[class_name] = nn.Embedding(vocab_size + use_cfg_embedding, n_embed)

        self.t_embedder = TimestepEmbedder(n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, n_embed), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=True,  # this is only true, cause we have time
                    elementwise_affine=False,
                )
                for _ in range(n_layer)
            ]
        )

        self.n_embed = n_embed
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_embed_input, n_embed, bias=bias)
        self.final_layer = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)

        # Initialize weights (timestep MLP, pos_embed, class embeddings, adaLN layers, output layer)
        self.initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
        force_drop_ids: bool | None = None,
    ) -> torch.Tensor:
        # Default: apply dropout during training, not during eval
        if force_drop_ids is None:
            force_drop_ids = self.training
        t_embedding = self.t_embedder(t).unsqueeze(1)

        condition_embedding: torch.Tensor = self._get_condition_embedding(condition, force_drop_ids)

        if condition_embedding is not None:
            t_embedding = t_embedding + condition_embedding

        x = self.input_proj(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x=x, condition=t_embedding)

        x = self.final_layer(x, t_embedding)
        return x

    def forward_with_cfg_joint(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        cfg_scale: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Classifier-free guidance with additive conditioning strategy.

        During sampling:
        1. Get unconditional prediction (all classes = null tokens)
        2. For each condition class, add its guidance independently

        This matches the training strategy where only one class is active at a time.
        """
        batch_size = len(x)

        # Create unconditional condition (all null tokens) across all classes
        uncond_condition = {}
        for class_name in self.class_vocab_sizes.keys():
            null_token = self.class_vocab_sizes[class_name]
            uncond_condition[class_name] = torch.full((batch_size,), null_token, device=x.device, dtype=torch.long)

        # Get unconditional prediction (all null tokens)
        uncond_out = self.forward(x, t, uncond_condition, force_drop_ids=False)
        guided_out = uncond_out.clone()

        # Apply guidance for each condition class independently
        if condition is not None and cfg_scale is not None:
            cond_out = self.forward(x, t, condition, force_drop_ids=False)
            guided_out += (
                cfg_scale["cell_line"] * (cond_out - uncond_out)
                # cfg_scale["cell_type"] * (cond_out - uncond_out)
            )  # TODO since we are joining the cell type and cytokine, we need only one scale term, for readability we should change it later to a better name

        return guided_out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        cfg_scale: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Efficient CFG sampling: first half unconditional, second half conditional."""
        batch_size = x.shape[0]
        len_half = batch_size // 2

        # Create unconditional condition (all null tokens) across all classes
        uncond_condition = {}
        for class_name in self.class_vocab_sizes.keys():
            null_token = self.class_vocab_sizes[class_name]
            uncond_condition[class_name] = torch.full((batch_size,), null_token, device=x.device, dtype=torch.long)

        uncond_out = self.forward(x, t, uncond_condition, force_drop_ids=False)

        # Split references
        uncond_out_half = uncond_out[:len_half]
        _cond_out_half = uncond_out[len_half:]
        cond_out_half = _cond_out_half.clone()

        # If no condition or scales, this leaves cond_out_half == base_half
        if condition is not None and cfg_scale is not None:
            x_half, t_half = x[len_half:], t[len_half:]

            if self.condition_strategy == "joint":
                # For joint condition strategy, pass all conditions together and use average scale
                full_condition_half = {k: v[len_half:] for k, v in condition.items()}
                cond_pred_half = self.forward(x_half, t_half, full_condition_half, force_drop_ids=False)
                avg_scale = sum(cfg_scale.values()) / len(cfg_scale)
                cond_out_half += avg_scale * (cond_pred_half - _cond_out_half)
            else:
                # For mutually_exclusive strategy, iterate per-class
                for class_name, scale in cfg_scale.items():
                    # Build per-class condition dict for the second half only
                    single_condition_half = {class_name: condition[class_name][len_half:]}
                    cond_pred_half = self.forward(x_half, t_half, single_condition_half, force_drop_ids=False)
                    cond_out_half += scale * (cond_pred_half - _cond_out_half)

        return torch.cat([uncond_out_half, cond_out_half], dim=0)

    def _get_condition_embedding(self, condition: dict[str, torch.Tensor], force_drop_ids: bool = True) -> torch.Tensor:
        batch_size = next(iter(condition.values())).shape[0]
        device = next(iter(condition.values())).device

        if self.condition_strategy == "joint":
            return self._get_joint_condition_embedding(condition, batch_size, device)
        else:  # Default to "mutually_exclusive"
            return self._get_mutually_exclusive_condition_embedding(condition, batch_size, device, force_drop_ids)

    def _get_mutually_exclusive_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device, force_drop_ids: bool = True
    ) -> torch.Tensor:
        """During training: randomly select one condition class per batch and apply CFG dropout."""
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]

        selected_class_idx = torch.randint(0, len(available_classes), (), device=device)

        # Optional per-sample dropout mask (device)
        if not self.training:
            assert not force_drop_ids, "force_drop_ids must be False when not training"

        if force_drop_ids:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())
        for class_name in class_names:
            null_token = self.class_vocab_sizes[class_name]
            if class_name in available_classes:
                # position of class_name in available_classes (python-level only; constant across graph)
                i = available_classes.index(class_name)
                # broadcast over batch
                is_selected_b = (selected_class_idx == i).expand(batch_size)

                cond_vals = condition[class_name]
                null_vals = torch.full_like(cond_vals, null_token)
                if force_drop_ids:
                    cond_or_null = torch.where(drop_mask, null_vals, cond_vals)
                    final_vals = torch.where(is_selected_b, cond_or_null, null_vals)
                else:
                    final_vals = torch.where(is_selected_b, cond_vals, null_vals)
            else:
                final_vals = torch.full((batch_size,), null_token, device=device, dtype=torch.long)

            class_emb = self.class_embeddings[class_name](final_vals)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_joint_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """During training: use all available conditions jointly with independent CFG dropout."""
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        # Apply CFG dropout independently to each condition class
        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())

        if self.training:
            drop_mask = (
                torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            )  # fixed masking for both types cells and cytokines
        else:
            drop_mask = torch.ones(batch_size, device=device) < 0  # no masking during inference

        for class_name in class_names:
            # Use the condition values with independent CFG dropout
            condition_values = condition[class_name]
            null_token = self.class_vocab_sizes[class_name]
            final_condition_values = torch.where(drop_mask, null_token, condition_values)
            class_emb = self.class_embeddings[class_name](final_condition_values)

            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def initialize_weights(self):
        # Initialize transformer layers with xavier uniform
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.n_embed, self.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding tables
        for embedding in self.class_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            if block.adaln_modulation[-1].bias is not None:
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        if self.final_layer.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)


import math

class CrossAttentionBlock_cell(nn.Module):
    """
    RNA/ATAC 双向通道交叉注意力块。

    该模块借鉴 scDiffusion-X 的 CrossAttentionBlock_cell：RNA latent
    以自身为 query、ATAC 为 key/value 更新 RNA；ATAC 分支反向执行同样
    的注意力更新。输入输出保持 ``(B, D, M)``，其中 ``D`` 是通道维，
    ``M`` 是诱导点/latent token 数。
    """
    def __init__(self, channels: int, feature_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim

        self.q_rna = nn.Linear(1, self.feature_dim, bias=False)
        self.k_rna = nn.Linear(1, self.feature_dim, bias=False)
        self.v_rna = nn.Linear(1, self.feature_dim, bias=False)

        self.q_atac = nn.Linear(1, self.feature_dim, bias=False)
        self.k_atac = nn.Linear(1, self.feature_dim, bias=False)
        self.v_atac = nn.Linear(1, self.feature_dim, bias=False)

        self.trans_rna = nn.Linear(self.feature_dim, 1, bias=False)
        self.trans_atac = nn.Linear(self.feature_dim, 1, bias=False)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video is RNA latent shape (B, D, M)
        # audio is ATAC latent shape (B, D, M)
        # We project along the last dimension (size 1) for each channel token
        video = video.transpose(-1, -2).unsqueeze(-1)  # (B, M, D, 1)
        audio = audio.transpose(-1, -2).unsqueeze(-1)  # (B, M, D, 1)

        q_rna = self.q_rna(video)
        k_rna = self.k_rna(video)
        v_rna = self.v_rna(video)

        q_atac = self.q_atac(audio)
        k_atac = self.k_atac(audio)
        v_atac = self.v_atac(audio)

        scale = math.sqrt(self.feature_dim)
        # matmul along D (channel tokens) -> shape: (B, M, D, D)
        att_rna_atac = torch.softmax(torch.matmul(q_rna, k_atac.transpose(-1, -2)) / scale, dim=-1)
        rna_new = torch.matmul(att_rna_atac, v_atac)

        att_atac_rna = torch.softmax(torch.matmul(q_atac, k_rna.transpose(-1, -2)) / scale, dim=-1)
        atac_new = torch.matmul(att_atac_rna, v_rna)

        video = video + self.trans_rna(rna_new)
        audio = audio + self.trans_atac(atac_new)

        video = video.squeeze(-1).transpose(-1, -2)
        audio = audio.squeeze(-1).transpose(-1, -2)

        return video, audio


class MultimodalDiT(nn.Module):
    """
    面向 Flow Matching 的 RNA+ATAC 多模态 DiT 去噪器。

    与原始单模态 DiT 不同，这里保留 RNA 和 ATAC 两条 latent stream。
    每一层先分别做 AdaLN Transformer block，再通过双向通道交叉注意力
    让两个模态交换信息，输出两个 velocity/noise 预测张量。
    """
    _DIRECTION_TO_ID = {"joint": 0, "rna_to_atac": 1, "atac_to_rna": 2}

    def __init__(
        self,
        n_embed: int,
        n_embed_input: int,
        n_layer: int,
        n_head: int,
        seq_len: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        class_vocab_sizes: dict[str, int],
        cfg_dropout_prob: float = 0.1,
        condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive",
        feature_dim: int = 64,
        backbone: Literal["flat", "uvit"] = "flat",
    ):
        super().__init__()
        if backbone not in {"flat", "uvit"}:
            raise ValueError("backbone must be either 'flat' or 'uvit'")
        if backbone == "uvit" and (n_layer < 2 or n_layer % 2 != 0):
            raise ValueError("U-ViT backbone requires an even n_layer >= 2")
        self.class_vocab_sizes = class_vocab_sizes
        self.cfg_dropout_prob = cfg_dropout_prob
        self.condition_strategy = condition_strategy
        self.backbone = backbone
        self.class_embeddings = nn.ModuleDict()
        for class_name, vocab_size in class_vocab_sizes.items():
            use_cfg_embedding = int(cfg_dropout_prob > 0)
            self.class_embeddings[class_name] = nn.Embedding(vocab_size + use_cfg_embedding, n_embed)

        self.t_embedder = TimestepEmbedder(n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, n_embed), requires_grad=False)
        self.n_embed = n_embed
        self.seq_len = seq_len

        self.input_proj_rna = nn.Linear(n_embed_input, n_embed, bias=bias)
        self.input_proj_atac = nn.Linear(n_embed_input, n_embed, bias=bias)

        def make_dual_block() -> nn.ModuleDict:
            return nn.ModuleDict({
                "rna_block": Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=True,
                    elementwise_affine=False,
                ),
                "atac_block": Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=True,
                    elementwise_affine=False,
                ),
                "cross_attn": CrossAttentionBlock_cell(
                    channels=n_embed,
                    feature_dim=feature_dim,
                )
            })

        # flat 模式完全保留旧版 MultimodalDiT 的顺序堆叠，保证旧 checkpoint 可加载。
        if self.backbone == "flat":
            self.blocks = nn.ModuleList([make_dual_block() for _ in range(n_layer)])
            self.encoder_blocks = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            self.rna_skip_projs = nn.ModuleList()
            self.atac_skip_projs = nn.ModuleList()
            self.direction_embedding = None
        else:
            # U-ViT 模式把双流 DiT 切成 encoder/decoder 两段。
            # encoder 的浅层表征会通过 skip connection 回流到 decoder，
            # 使 RNA/ATAC 的局部细节不用完全挤过最深层的压缩通道。
            n_encoder = n_layer // 2
            n_decoder = n_layer // 2
            self.blocks = nn.ModuleList()
            self.encoder_blocks = nn.ModuleList([make_dual_block() for _ in range(n_encoder)])
            self.decoder_blocks = nn.ModuleList([make_dual_block() for _ in range(n_decoder)])
            self.rna_skip_projs = nn.ModuleList(
                [nn.Linear(2 * n_embed, n_embed, bias=False) for _ in range(n_decoder)]
            )
            self.atac_skip_projs = nn.ModuleList(
                [nn.Linear(2 * n_embed, n_embed, bias=False) for _ in range(n_decoder)]
            )
            # 显式方向嵌入让同一套 U-ViT 区分 joint / RNA->ATAC / ATAC->RNA 三类流任务。
            self.direction_embedding = nn.Embedding(len(self._DIRECTION_TO_ID), n_embed)

        self.final_layer_rna = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)
        self.final_layer_atac = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)

        self.initialize_weights()

    def _iter_block_dicts(self):
        if self.backbone == "flat":
            yield from self.blocks
        else:
            yield from self.encoder_blocks
            yield from self.decoder_blocks

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Sin-cos positional embedding
        pos_embed = get_1d_sincos_pos_embed(self.n_embed, self.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Class embedding init
        for embedding in self.class_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)
        if isinstance(self.direction_embedding, nn.Embedding):
            nn.init.normal_(self.direction_embedding.weight, std=0.02)

        # Timestep MLPs
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block_dict in self._iter_block_dicts():
            nn.init.constant_(block_dict["rna_block"].adaln_modulation[-1].weight, 0)
            if block_dict["rna_block"].adaln_modulation[-1].bias is not None:
                nn.init.constant_(block_dict["rna_block"].adaln_modulation[-1].bias, 0)
            nn.init.constant_(block_dict["atac_block"].adaln_modulation[-1].weight, 0)
            if block_dict["atac_block"].adaln_modulation[-1].bias is not None:
                nn.init.constant_(block_dict["atac_block"].adaln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_rna.adaln_modulation[-1].weight, 0)
        if self.final_layer_rna.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer_rna.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_rna.linear.weight, 0)
        if self.final_layer_rna.linear.bias is not None:
            nn.init.constant_(self.final_layer_rna.linear.bias, 0)

        nn.init.constant_(self.final_layer_atac.adaln_modulation[-1].weight, 0)
        if self.final_layer_atac.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer_atac.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_atac.linear.weight, 0)
        if self.final_layer_atac.linear.bias is not None:
            nn.init.constant_(self.final_layer_atac.linear.bias, 0)

    def _get_condition_embedding(
        self,
        condition: dict[str, torch.Tensor] | None,
        force_drop_ids: bool = True,
    ) -> torch.Tensor:
        if not condition:
            raise ValueError("_get_condition_embedding requires batch metadata; call _empty_condition_embedding instead")
        batch_size = next(iter(condition.values())).shape[0]
        device = next(iter(condition.values())).device
        if self.condition_strategy == "joint":
            return self._get_joint_condition_embedding(condition, batch_size, device)
        else:
            return self._get_mutually_exclusive_condition_embedding(condition, batch_size, device, force_drop_ids)

    def _empty_condition_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """无条件训练时使用的零条件嵌入，保持与 CFG 条件接口兼容。"""
        return torch.zeros(batch_size, 1, self.n_embed, device=device)

    def _get_mutually_exclusive_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device, force_drop_ids: bool = True
    ) -> torch.Tensor:
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        selected_class_idx = torch.randint(0, len(available_classes), (), device=device)
        if force_drop_ids:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob

        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())
        for class_name in class_names:
            null_token = self.class_vocab_sizes[class_name]
            if class_name in available_classes:
                i = available_classes.index(class_name)
                is_selected_b = (selected_class_idx == i).expand(batch_size)
                cond_vals = condition[class_name]
                null_vals = torch.full_like(cond_vals, null_token)
                if force_drop_ids:
                    cond_or_null = torch.where(drop_mask, null_vals, cond_vals)
                    final_vals = torch.where(is_selected_b, cond_or_null, null_vals)
                else:
                    final_vals = torch.where(is_selected_b, cond_vals, null_vals)
            else:
                final_vals = torch.full((batch_size,), null_token, device=device, dtype=torch.long)

            class_emb = self.class_embeddings[class_name](final_vals)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_joint_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        available_classes = [name for name in sorted(self.class_vocab_sizes.keys()) if name in condition]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())

        if self.training:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
        else:
            drop_mask = torch.ones(batch_size, device=device) < 0

        for class_name in class_names:
            condition_values = condition[class_name]
            null_token = self.class_vocab_sizes[class_name]
            final_condition_values = torch.where(drop_mask, null_token, condition_values)
            class_emb = self.class_embeddings[class_name](final_condition_values)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_direction_embedding(
        self,
        mode: Literal["joint", "rna_to_atac", "atac_to_rna"] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(self.direction_embedding, nn.Embedding):
            return torch.zeros(batch_size, 1, self.n_embed, device=device)
        direction = "joint" if mode is None else mode
        if direction not in self._DIRECTION_TO_ID:
            raise ValueError(f"Unsupported multimodal flow direction: {direction}")
        direction_id = self._DIRECTION_TO_ID[direction]
        ids = torch.full((batch_size,), direction_id, device=device, dtype=torch.long)
        return self.direction_embedding(ids).unsqueeze(1)

    def _apply_dual_block(
        self,
        h_rna: torch.Tensor,
        h_atac: torch.Tensor,
        t_embedding: torch.Tensor,
        block_dict: nn.ModuleDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_rna = block_dict["rna_block"](x=h_rna, condition=t_embedding)
        h_atac = block_dict["atac_block"](x=h_atac, condition=t_embedding)

        # 双向通道交叉注意力：把通道维当作序列维，让 RNA/ATAC 在 latent 通道上互相借信息。
        video = h_rna.transpose(-1, -2)
        audio = h_atac.transpose(-1, -2)
        video, audio = block_dict["cross_attn"](video, audio)
        h_rna = video.transpose(-1, -2)
        h_atac = audio.transpose(-1, -2)
        return h_rna, h_atac

    def forward(
        self,
        x_rna: torch.Tensor,
        x_atac: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        force_drop_ids: bool | None = None,
        mode: Literal["joint", "rna_to_atac", "atac_to_rna"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if force_drop_ids is None:
            force_drop_ids = self.training
        t_embedding = self.t_embedder(t).unsqueeze(1)

        if condition:
            condition_embedding = self._get_condition_embedding(condition, force_drop_ids)
        else:
            condition_embedding = self._empty_condition_embedding(x_rna.shape[0], x_rna.device)
        if condition_embedding is not None:
            t_embedding = t_embedding + condition_embedding
        if self.backbone == "uvit":
            t_embedding = t_embedding + self._get_direction_embedding(mode, x_rna.shape[0], x_rna.device)

        # Project inputs
        h_rna = self.input_proj_rna(x_rna) + self.pos_embed
        h_atac = self.input_proj_atac(x_atac) + self.pos_embed

        if self.backbone == "flat":
            # 旧版扁平 DiT：顺序通过所有双流 block。
            for block_dict in self.blocks:
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)
        else:
            # U-ViT：encoder 段保存浅层 RNA/ATAC 表征，decoder 段逆序取回并融合。
            rna_skips: list[torch.Tensor] = []
            atac_skips: list[torch.Tensor] = []
            for block_dict in self.encoder_blocks:
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)
                rna_skips.append(h_rna)
                atac_skips.append(h_atac)

            for idx, block_dict in enumerate(self.decoder_blocks):
                h_rna_skip = rna_skips[-(idx + 1)]
                h_atac_skip = atac_skips[-(idx + 1)]
                h_rna = self.rna_skip_projs[idx](torch.cat([h_rna, h_rna_skip], dim=-1))
                h_atac = self.atac_skip_projs[idx](torch.cat([h_atac, h_atac_skip], dim=-1))
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)

        # Output final velocity/noise predictions
        out_rna = self.final_layer_rna(h_rna, t_embedding)
        out_atac = self.final_layer_atac(h_atac, t_embedding)

        return out_rna, out_atac
