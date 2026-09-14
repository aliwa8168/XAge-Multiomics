import torch
import torch.nn as nn
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI

from scflowdiff.core.layers import Block, InputTransformerAE
from scflowdiff.core.nnets import Decoder, Encoder
from scflowdiff.core.stochastic_layers import (
    BernoulliTransformerLayer,
    NegativeBinomialTransformerLayer,
)


class TransformerAE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        decoder_head: NegativeBinomialTransformerLayer,
        input_layer: InputTransformerAE,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_head = decoder_head
        self.input_layer = input_layer

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        genes_counts_embedding = self.input_layer(
            counts_subset,
            genes_subset,
        )  # B, S, E
        h_z = self.encoder(genes_counts_embedding)  # B, M, E
        genes_for_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )  # B, S, E
        h_x = self.decoder(h_z, genes_for_decoder)  # B, S, E
        head_name = self.decoder_head.__class__.__name__
        if head_name == "GaussianTransformerLayer":
            mu = self.decoder_head(h_x, genes, library_size)
            params = {"mu": mu}
        else:
            out = self.decoder_head(h_x, genes, library_size)
            if isinstance(out, tuple) and len(out) == 2:
                params = {"mu": out[0], "theta": out[1]}
            else:
                raise ValueError(f"Unsupported decoder_head output for {head_name}: {type(out)}")
        return params, h_z

    def encode(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        genes_counts_embedding = self.input_layer(
            counts_subset if counts_subset is not None else counts,
            genes_subset if genes_subset is not None else genes,
        )
        return self.encoder(genes_counts_embedding)

    def decode(
        self,
        z: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
    ) -> torch.distributions.Distribution:
        genes_for_decoder = (
            genes if isinstance(self.decoder.gene_embedding, nn.Embedding) else self.input_layer.gene_embedding(genes)
        )
        h_x = self.decoder(z, genes_for_decoder, condition)
        head_name = self.decoder_head.__class__.__name__
        if head_name == "GaussianTransformerLayer":
            mu = self.decoder_head(h_x, genes, library_size)
            return Normal(mu, torch.ones_like(mu))
        mu, theta = self.decoder_head(h_x, genes, library_size)
        return NegativeBinomialSCVI(mu=mu, theta=theta)


class MultimodalTransformerAE(nn.Module):
    """Deterministic two-branch Transformer AE for paired RNA and ATAC.

    RNA and ATAC are encoded by independent MCAB-style encoder stacks. During
    decoding, gene tokens query ``Z_rna`` and peak tokens query ``Z_atac`` via
    independent cross-attention decoders. This preserves modality-specific
    biological structure while allowing the training module to optimize a joint
    likelihood.
    """

    def __init__(
        self,
        *,
        rna_encoder: Encoder,
        rna_decoder: Decoder,
        rna_decoder_head: NegativeBinomialTransformerLayer,
        rna_input_layer: InputTransformerAE,
        atac_encoder: Encoder,
        atac_decoder: Decoder,
        atac_decoder_head: BernoulliTransformerLayer,
        atac_input_layer: InputTransformerAE,
        multimodal: str = "multimodal",
        encoder_multimodal_joint_layers: int | None = None,
        n_head: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        norm_layer: str = "layernorm",
        multiple_of: int = 4,
        layernorm_eps: float = 1e-6,
    ):
        super().__init__()
        if multimodal not in {"RNA", "multimodal"}:
            raise ValueError("multimodal must be either 'RNA' or 'multimodal'")
        self.multimodal = multimodal
        self.rna_encoder = rna_encoder
        self.rna_decoder = rna_decoder
        self.rna_decoder_head = rna_decoder_head
        self.rna_input_layer = rna_input_layer
        self.atac_encoder = atac_encoder
        self.atac_decoder = atac_decoder
        self.atac_decoder_head = atac_decoder_head
        self.atac_input_layer = atac_input_layer
        
        self.encoder_multimodal_joint_layers = encoder_multimodal_joint_layers
        if encoder_multimodal_joint_layers is not None and encoder_multimodal_joint_layers > 0:
            self.joint_encoder = nn.ModuleList([
                Block(
                    n_embed=rna_encoder.latent_embedding,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                )
                for _ in range(encoder_multimodal_joint_layers)
            ])
        else:
            self.joint_encoder = None

    @staticmethod
    def _decoder_queries(
        decoder: Decoder,
        input_layer: InputTransformerAE,
        tokens: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        if isinstance(decoder.gene_embedding, nn.Embedding):
            return tokens
        return input_layer.token_embedding(tokens.long(), modality=modality)  # type: ignore[arg-type]

    def _forward_rna(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        rna_embedding = self.rna_input_layer(
            counts_subset if counts_subset is not None else counts,
            genes_subset if genes_subset is not None else genes,
            modality="rna",
        )
        rna_mask = genes_subset.long() != 0 if genes_subset is not None else None
        h_z = self.rna_encoder(rna_embedding, key_padding_mask=rna_mask)
        decoder_queries = self._decoder_queries(self.rna_decoder, self.rna_input_layer, genes, "rna")
        h_x = self.rna_decoder(h_z, decoder_queries)
        out = self.rna_decoder_head(h_x, genes, library_size)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise ValueError(f"Unsupported RNA decoder head output: {type(out)}")
        params = {"mu": out[0], "theta": out[1]}
        return params, h_z

    def _forward_atac(
        self,
        values: torch.Tensor,
        peaks: torch.Tensor,
        values_subset: torch.Tensor | None = None,
        peaks_subset: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        atac_embedding = self.atac_input_layer(
            values_subset if values_subset is not None else values,
            peaks_subset if peaks_subset is not None else peaks,
            modality="atac",
        )
        atac_mask = peaks_subset.long() != 0 if peaks_subset is not None else None
        h_z = self.atac_encoder(atac_embedding, key_padding_mask=atac_mask)
        decoder_queries = self._decoder_queries(self.atac_decoder, self.atac_input_layer, peaks, "atac")
        h_x = self.atac_decoder(h_z, decoder_queries)
        probs = self.atac_decoder_head(h_x, peaks, None)
        return {"probs": probs}, h_z

    def _apply_joint_encoder(
        self,
        h_z_rna: torch.Tensor,
        h_z_atac: torch.Tensor,
    ) -> torch.Tensor:
        """用 joint self-attention 建模 RNA/ATAC 的联合诱导点隐空间。

        Joint multimodal autoencoders concatenate the two modality-specific
        latent token sets into ``(B, 2M, D)`` before joint attention. Here the
        RNA/ATAC branches remain distinct while
        ``encoder_multimodal_joint_layers`` supplies the shared decoder memory.
        """
        if self.joint_encoder is None:
            raise RuntimeError("Joint encoder was requested but is not configured")
        z_joint = torch.cat([h_z_rna, h_z_atac], dim=1)
        for layer in self.joint_encoder:
            z_joint = layer(z_joint)
        return z_joint

    @staticmethod
    def _split_joint_latent(
        z_joint: torch.Tensor,
        n_rna_tokens: int,
        n_atac_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = n_rna_tokens + n_atac_tokens
        if z_joint.shape[1] != expected:
            raise ValueError(
                f"Joint latent has {z_joint.shape[1]} tokens, expected {expected} "
                f"({n_rna_tokens} RNA + {n_atac_tokens} ATAC)"
            )
        return z_joint[:, :n_rna_tokens], z_joint[:, n_rna_tokens:]

    def forward(
        self,
        rna_counts: torch.Tensor,
        rna_genes: torch.Tensor,
        rna_library_size: torch.Tensor,
        atac_values: torch.Tensor | None = None,
        atac_peaks: torch.Tensor | None = None,
        rna_counts_subset: torch.Tensor | None = None,
        rna_genes_subset: torch.Tensor | None = None,
        atac_values_subset: torch.Tensor | None = None,
        atac_peaks_subset: torch.Tensor | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        # Encode RNA
        rna_embedding = self.rna_input_layer(
            rna_counts_subset if rna_counts_subset is not None else rna_counts,
            rna_genes_subset if rna_genes_subset is not None else rna_genes,
            modality="rna",
        )
        rna_mask = rna_genes_subset.long() != 0 if rna_genes_subset is not None else None
        h_z_rna = self.rna_encoder(rna_embedding, key_padding_mask=rna_mask)
        
        # Encode ATAC if multimodal
        if self.multimodal == "multimodal":
            if atac_values is None or atac_peaks is None:
                raise ValueError("ATAC values and peak tokens are required when multimodal == 'multimodal'")
            atac_embedding = self.atac_input_layer(
                atac_values_subset if atac_values_subset is not None else atac_values,
                atac_peaks_subset if atac_peaks_subset is not None else atac_peaks,
                modality="atac",
            )
            atac_mask = atac_peaks_subset.long() != 0 if atac_peaks_subset is not None else None
            h_z_atac = self.atac_encoder(atac_embedding, key_padding_mask=atac_mask)
        else:
            h_z_atac = None

        if self.multimodal == "multimodal" and self.joint_encoder is not None:
            z_joint = self._apply_joint_encoder(h_z_rna, h_z_atac)
            z_rna_joint, z_atac_joint = self._split_joint_latent(
                z_joint, h_z_rna.shape[1], h_z_atac.shape[1]
            )

            # Joint self-attention 后恢复两个模态各自的 token 分支，再送入对应 decoder。
            rna_decoder_queries = self._decoder_queries(self.rna_decoder, self.rna_input_layer, rna_genes, "rna")
            h_x_rna = self.rna_decoder(z_rna_joint, rna_decoder_queries)
            out_rna = self.rna_decoder_head(h_x_rna, rna_genes, rna_library_size)
            
            atac_decoder_queries = self._decoder_queries(self.atac_decoder, self.atac_input_layer, atac_peaks, "atac")
            h_x_atac = self.atac_decoder(z_atac_joint, atac_decoder_queries)
            probs_atac = self.atac_decoder_head(h_x_atac, atac_peaks, None)
            
            params = {
                "rna": {"mu": out_rna[0], "theta": out_rna[1]},
                "atac": {"probs": probs_atac}
            }
            latents = {
                "rna": h_z_rna,
                "atac": h_z_atac,
                "joint": z_joint,
                "rna_joint": z_rna_joint,
                "atac_joint": z_atac_joint,
            }
            return params, latents
        else:
            # Original separate-latent logic
            params_rna, _ = self._forward_rna(
                rna_counts, rna_genes, rna_library_size, rna_counts_subset, rna_genes_subset
            )
            params = {"rna": params_rna}
            latents = {"rna": h_z_rna}
            
            if self.multimodal == "multimodal":
                params_atac, _ = self._forward_atac(
                    atac_values, atac_peaks, atac_values_subset, atac_peaks_subset
                )
                params["atac"] = params_atac
                latents["atac"] = h_z_atac
            return params, latents

    def encode(
        self,
        rna_counts: torch.Tensor,
        rna_genes: torch.Tensor,
        atac_values: torch.Tensor | None = None,
        atac_peaks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_rna = self.rna_encoder(self.rna_input_layer(rna_counts, rna_genes, modality="rna"))
        latents = {"rna": z_rna}
        if self.multimodal == "multimodal":
            if atac_values is None or atac_peaks is None:
                raise ValueError("ATAC values and peak tokens are required for multimodal encoding")
            z_atac = self.atac_encoder(self.atac_input_layer(atac_values, atac_peaks, modality="atac"))
            latents["atac"] = z_atac
            
            if self.joint_encoder is not None:
                latents["joint"] = self._apply_joint_encoder(z_rna, z_atac)
        return latents

    def decode_rna(
        self,
        z_rna: torch.Tensor,
        rna_genes: torch.Tensor,
        rna_library_size: torch.Tensor,
        z_atac_source: torch.Tensor | None = None,
    ) -> NegativeBinomialSCVI:
        if self.joint_encoder is not None and z_atac_source is not None:
            if z_rna.shape[0] != z_atac_source.shape[0] or z_rna.shape[-1] != z_atac_source.shape[-1]:
                raise ValueError(
                    "RNA target latent and ATAC source latent must share batch size and embedding dimension, "
                    f"got {tuple(z_rna.shape)} and {tuple(z_atac_source.shape)}"
                )
            z_joint = self._apply_joint_encoder(z_rna, z_atac_source)
            z_input, _ = self._split_joint_latent(
                z_joint, z_rna.shape[1], z_atac_source.shape[1]
            )
        else:
            z_input = z_rna
        decoder_queries = self._decoder_queries(self.rna_decoder, self.rna_input_layer, rna_genes, "rna")
        h_x = self.rna_decoder(z_input, decoder_queries)
        out = self.rna_decoder_head(h_x, rna_genes, rna_library_size)
        if isinstance(out, tuple) and len(out) == 2:
            mu, theta = out
            return NegativeBinomialSCVI(mu=mu, theta=theta)
        raise ValueError(f"Unsupported RNA decoder head output: {type(out)}")

    def decode_atac(
        self,
        z_atac: torch.Tensor,
        atac_peaks: torch.Tensor,
        z_rna_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.joint_encoder is not None and z_rna_source is not None:
            if z_atac.shape[0] != z_rna_source.shape[0] or z_atac.shape[-1] != z_rna_source.shape[-1]:
                raise ValueError(
                    "ATAC target latent and RNA source latent must share batch size and embedding dimension, "
                    f"got {tuple(z_atac.shape)} and {tuple(z_rna_source.shape)}"
                )
            z_joint = self._apply_joint_encoder(z_rna_source, z_atac)
            _, z_input = self._split_joint_latent(
                z_joint, z_rna_source.shape[1], z_atac.shape[1]
            )
        else:
            z_input = z_atac
        decoder_queries = self._decoder_queries(self.atac_decoder, self.atac_input_layer, atac_peaks, "atac")
        h_x = self.atac_decoder(z_input, decoder_queries)
        return self.atac_decoder_head(h_x, atac_peaks, None)

