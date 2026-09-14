import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution

from scflowdiff.core.distributions import log_nb_positive, log_zinb_positive
from scflowdiff.core.layers import NORM_LAYERS


###########################
#     GAUSSIAN LAYERS     #
###########################
class GaussianTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
    ):
        super().__init__()
        if n_embed is None:
            raise ValueError("GaussianTransformerLayer requires n_embed (got None)")
        self.ln = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.params = nn.Linear(n_embed, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        genes: torch.Tensor | None = None,
        library_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ln(x)
        mu = self.params(x)
        return mu.squeeze(-1)


class GaussianLinearLayer(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_latent: int,
    ):
        super().__init__()
        self.loc = nn.Linear(n_hidden, n_latent, bias=True)
        self.scale = nn.Linear(n_hidden, n_latent, bias=True)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        # location
        loc = self.loc(x)
        # scale
        log_scale = self.scale(x)
        log_scale = nn.functional.hardtanh(log_scale, min_val=-7.0, max_val=5.0)
        scale = torch.exp(log_scale)
        return dist.Normal(loc, scale)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x)
        return distribution.rsample()

    def log_prob(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        if (loc is None) or (scale is None):
            distribution = self.forward(x)
        else:
            distribution = dist.Normal(loc, scale)
        log_p = distribution.log_prob(x)
        return log_p

    def loss(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        return self.log_prob(x, loc, scale)


############################
#     NBinomial LAYERS     #
############################
class NegativeBinomialTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps_: float = 1e-6,
        t: float = 1.0,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        if shared_theta:
            self.theta = nn.Embedding(n_genes + 1, 1)
            torch.nn.init.ones_(self.theta.weight)
            self.params = nn.Linear(n_embed, 1, bias=True)
        else:
            self.theta = None
            self.params = nn.Linear(n_embed, 2, bias=True)

        self.eps_ = eps_
        self.t = t

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.theta, nn.Embedding):
            mu = self.params(counts)
            theta = self.theta(genes.long())
        else:
            params = self.params(counts)
            mu, theta = torch.chunk(params, 2, dim=-1)
        mu, theta = mu.squeeze(-1), torch.exp(theta).squeeze(-1)
        mu = nn.functional.softmax(mu / self.t, dim=1) * library_size
        return mu, theta

    def log_prob(
        self,
        target: torch.Tensor,
        decoder_states: torch.Tensor,
        genes: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        mu, theta = self.forward(decoder_states, genes, total_counts)
        return log_nb_positive(target, mu, theta, eps=self.eps_)


class ZeroInflatedNegativeBinomialTransformerLayer(nn.Module):
    """ZINB likelihood head for sparse scRNA-seq reconstruction.

    The decoder hidden state is mapped to NB mean logits plus a zero-inflation
    logit. ``mu`` is softmax-normalized over the queried genes and scaled by the
    observed cell library size, while ``theta`` is constrained positive.
    """

    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps_: float = 1e-6,
        t: float = 1.0,
    ):
        super().__init__()
        if n_embed is None:
            raise ValueError("ZeroInflatedNegativeBinomialTransformerLayer requires n_embed (got None)")
        self.shared_theta = shared_theta
        self.ln = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        if shared_theta:
            self.theta = nn.Embedding(n_genes + 1, 1)
            torch.nn.init.ones_(self.theta.weight)
            self.params = nn.Linear(n_embed, 2, bias=True)
        else:
            self.theta = None
            self.params = nn.Linear(n_embed, 3, bias=True)
        self.eps_ = eps_
        self.t = t

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.ln(counts)
        if isinstance(self.theta, nn.Embedding):
            mu_logits, zi_logits = torch.chunk(self.params(hidden), 2, dim=-1)
            theta = self.theta(genes.long())
        else:
            mu_logits, theta, zi_logits = torch.chunk(self.params(hidden), 3, dim=-1)
        mu_logits = mu_logits.squeeze(-1)
        theta = torch.exp(theta).squeeze(-1).clamp_min(self.eps_)
        zi_logits = zi_logits.squeeze(-1)
        mu = nn.functional.softmax(mu_logits / self.t, dim=1) * library_size
        return mu, theta, zi_logits

    def log_prob(
        self,
        target: torch.Tensor,
        decoder_states: torch.Tensor,
        genes: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        mu, theta, zi_logits = self.forward(decoder_states, genes, total_counts)
        return log_zinb_positive(target, mu, theta, zi_logits, eps=self.eps_, pi_is_logits=True)


class NegativeBinomialLinearLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        n_hidden: int,
        shared_theta: bool = False,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        self.mu = nn.Linear(n_hidden, n_genes, bias=True)
        if self.shared_theta:
            self.theta: nn.Parameter | nn.Linear = nn.Parameter(torch.ones(n_genes), requires_grad=True)
        else:
            self.theta: nn.Linear = nn.Linear(n_hidden, n_genes, bias=True)
        self.softplus = nn.Softplus()

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> Distribution:
        mu = self.mu(counts)
        if isinstance(self.theta, nn.Parameter):
            theta = self.softplus(self.theta)
        else:
            theta = self.softplus(self.theta(counts))
        mu = nn.functional.softmax(mu, dim=1)
        mu = mu * library_size
        return NegativeBinomialSCVI(mu=mu, theta=theta)

    def log_prob(self, x: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x, total_counts)
        return distribution.log_prob(x)


############################
#     BERNOULLI LAYERS     #
############################
class BernoulliTransformerLayer(nn.Module):
    """Bernoulli likelihood head for binary scATAC-seq peak reconstruction.

    The transformer decoder produces one hidden vector per queried peak. This
    layer maps each hidden vector to a peak-open probability and exposes a
    ``log_prob`` helper whose output has the same shape as the target matrix.
    Keeping this head separate from the RNA NB head makes the statistical
    assumption explicit: RNA counts are overdispersed counts, while ATAC peaks
    are modeled as binary accessibility events.
    """

    def __init__(
        self,
        *,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_embed is None:
            raise ValueError("BernoulliTransformerLayer requires n_embed (got None)")
        self.ln = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.logits = nn.Linear(n_embed, 1, bias=True)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        peaks: torch.Tensor | None = None,
        library_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return peak-open probabilities in ``[0, 1]`` with shape ``B x S``."""
        if x.ndim != 3:
            raise ValueError(f"Expected decoder states with shape (batch, seq, embed), got {tuple(x.shape)}")
        logits = self.logits(self.ln(x)).squeeze(-1)
        return torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

    def log_prob(
        self,
        target: torch.Tensor,
        probs: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Return Bernoulli log-probability, i.e. negative BCE, per peak by default."""
        if target.shape != probs.shape:
            raise ValueError(f"target shape {tuple(target.shape)} must match probs shape {tuple(probs.shape)}")
        target = target.float().clamp(min=0.0, max=1.0)
        probs = probs.float().clamp(min=self.eps, max=1.0 - self.eps)
        return -F.binary_cross_entropy(probs, target, reduction=reduction)
