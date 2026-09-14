from collections.abc import Callable

import torch


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: Callable[[torch.Tensor], torch.Tensor] = torch.log,
    lgamma_fn: Callable[[torch.Tensor], torch.Tensor] = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    log_fn: Callable[[torch.Tensor], torch.Tensor] = torch.log,
    lgamma_fn: Callable[[torch.Tensor], torch.Tensor] = torch.lgamma,
    pi_is_logits: bool = True,
) -> torch.Tensor:
    """Elementwise log likelihood under a zero-inflated negative binomial model.

    Parameters
    ----------
    x
        Observed counts.
    mu
        Mean of the negative binomial component.
    theta
        Inverse dispersion of the negative binomial component.
    pi
        Zero-inflation parameter. By default this is interpreted as logits,
        matching scVI/DCA-style heads that emit unconstrained ``zi_logits``.
        Set ``pi_is_logits=False`` to pass probabilities directly.
    eps
        Numerical stability constant.
    """
    nb_case = log_nb_positive(x=x, mu=mu, theta=theta, eps=eps, log_fn=log_fn, lgamma_fn=lgamma_fn)
    if pi_is_logits:
        zi_logits = pi
        log_pi = -torch.nn.functional.softplus(-zi_logits)
        log_one_minus_pi = -torch.nn.functional.softplus(zi_logits)
    else:
        pi_prob = pi.clamp(min=eps, max=1.0 - eps)
        log_pi = log_fn(pi_prob)
        log_one_minus_pi = log_fn(1.0 - pi_prob)

    nb_zero = log_nb_positive(
        x=torch.zeros_like(x),
        mu=mu,
        theta=theta,
        eps=eps,
        log_fn=log_fn,
        lgamma_fn=lgamma_fn,
    )
    zero_case = torch.logaddexp(log_pi, log_one_minus_pi + nb_zero)
    non_zero_case = log_one_minus_pi + nb_case
    return torch.where(x < eps, zero_case, non_zero_case)


def log_gaussian(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor | None = None,
    eps: float = 1e-8,
    log_fn: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """Gaussian-style reconstruction loss helper.

    - If ``sigma`` is provided: returns a Gaussian negative log-likelihood term
      (up to an additive constant) under Normal(mu, sigma).
    - If ``sigma`` is ``None``: returns an elementwise L2 loss (x - mu)^2.
    """
    if sigma is None:
        return (x - mu) ** 2

    sigma = sigma + eps
    return 0.5 * torch.pow((x - mu) / sigma, 2) + log_fn(sigma)
