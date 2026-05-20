from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .time_sampling import sample_s


def diffclf_loss_multi(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable,
    score_fn: Callable,
    log_q_fn: Callable,
    log_q_and_score_fn: Callable,
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    *,
    k: int = 2,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Diffusion classification loss (DiffCLF) adapted to JAX.
    - Samples k noise levels per sample.
    - Perturbs x0 to xt for each sampled time.
    - Forms all k x k (i, j) pairs and computes logits = log q_t_i(xt_j).
    - Computes multiclass NLL across time labels per sample and averages.
    - Uses log_q_and_score_fn for diagonal entries to compute DSM loss simultaneously.
    Args:
      weight: Coefficient for the classification term.
    Returns:
      (clf_loss, dsm_loss)
    """
    del score_fn  # not used for classification
    assert k >= 2, "k must be at least 2"

    BS = x.shape[0]
    data_shape = x.shape[1:]
    key, t_key, noise_key = jax.random.split(key, 3)

    # Sample k-1 times uniformly in (eps, 1.0] and concatenate the provided t
    ts_uniform = jax.random.uniform(t_key, (BS, k - 1, 1), minval=eps, maxval=1.0)
    ts = jnp.concatenate([ts_uniform, t.reshape(BS, 1, 1)], axis=1)

    # Perturb x0 -> xt using VP marginals
    x0_expanded = jnp.broadcast_to(x[:, None, ...], (BS, k, *data_shape))
    mean, std = sde.marginal_prob(x0_expanded, ts)
    noise = jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
    xt = mean + std * noise

    # Diagonal entries: i == j
    xt_diag = xt.reshape(BS * k, *data_shape)
    ts_diag = ts.reshape(BS * k, 1)
    if features is not None:
        features_diag = jnp.repeat(features[:, None, ...], k, axis=1)
        features_diag = features_diag.reshape(BS * k, *features_diag.shape[2:])
    else:
        features_diag = None

    logp_ii, score_ii = log_q_and_score_fn(xt_diag, features_diag, ts_diag)
    logp_ii = logp_ii.reshape(BS, k)

    # Compute DSM loss on the diagonal
    noise_flat = noise.reshape(BS * k, *noise.shape[2:])
    std_flat = std.reshape(BS * k, 1)
    if dsm_loss_fn is not None:
        dsm_loss = dsm_loss_fn(noise_flat, score_ii, std_flat, ts_diag)
    else:
        dsm_loss = jnp.array(0.0)

    # Off-diagonal entries: i != j
    i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    mask = i_idx != j_idx
    i_idx_off = i_idx[mask]
    j_idx_off = j_idx[mask]

    ts_off = ts[:, i_idx_off, :]
    xt_off = xt[:, j_idx_off, :]

    if features is not None:
        features_off = jnp.repeat(features[:, None, ...], k * (k - 1), axis=1)
        features_off = features_off.reshape(BS * k * (k - 1), *features_off.shape[2:])
    else:
        features_off = None

    ts_off_flat = ts_off.reshape(BS * k * (k - 1), 1)
    xt_off_flat = xt_off.reshape(BS * k * (k - 1), *xt_off.shape[2:])

    logp_off = log_q_fn(xt_off_flat, features_off, ts_off_flat)
    logp_off = logp_off.reshape(BS, k * (k - 1))

    # Reconstruct the full k x k matrix
    logits = jnp.zeros((BS, k, k), dtype=logp_ii.dtype)
    logits = logits.at[:, jnp.arange(k), jnp.arange(k)].set(logp_ii)
    logits = logits.at[:, i_idx_off, j_idx_off].set(logp_off)

    # For each j: lse over i
    lse = jax.scipy.special.logsumexp(logits, axis=1)  # (BS, k)

    # Per-sample multiclass NLL averaged over j, then batch-mean
    clf_per_sample = -(logp_ii - lse).mean(axis=1)  # (BS,)
    clf_loss = clf_per_sample.mean()  # scalar

    return weight * clf_loss, dsm_loss



def diffclf_loss_binary(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    *,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    DiffCLF binary loss (k=2 special case of DiffCLF, equivalent to TNCE with time_std=0).
    Samples a second time s uniformly from (eps, 1.0] (independent of t) and perturbs x0
    to both xt and xs using shared noise. Computes the binary contrastive loss via
    softplus(Z1) + softplus(Z2) where Z1, Z2 are log-ratio terms between cross and
    on-diagonal evaluations.
    Args:
      weight: Coefficient for the classification term.
    Returns:
      (weighted_clf_loss, dsm_loss)
    """
    del score_fn

    BS = x.shape[0]
    key, s_key, noise_key = jax.random.split(key, 3)

    # s = jax.random.uniform(s_key, shape=t.shape, minval=eps, maxval=1.0)
    t, s = sample_s(s_key, t, eps=eps, time_std=0.0)

    ts = jnp.concatenate([t, s], axis=0)
    x0 = jnp.concatenate([x, x], axis=0)
    if features is not None:
        features_ts = jnp.concatenate([features, features], axis=0)
    else:
        features_ts = None

    mean, std = sde.marginal_prob(x0, ts)

    # z = jax.random.normal(noise_key, (BS, 2, x.shape[-1]), dtype=x.dtype)
    z = jax.random.normal(noise_key, (BS, x.shape[-1]), dtype=x.dtype)
    noise = jnp.concatenate([z, z], axis=0)
    # noise = jax.random.normal(noise_key, x.shape, dtype=x.dtype)  # dependent noise
    # noise = jnp.concatenate([noise, noise], axis=0)
    xts = mean + noise * std
    xt = xts[:BS]
    xs = xts[BS:]

    if dsm_loss_fn is not None:
        logp_xts_ts, score_xts = log_q_and_score_fn(xts, features_ts, ts)
        dsm_loss = dsm_loss_fn(noise, score_xts, std, ts)
    else:
        logp_xts_ts = log_q_fn(xts, features_ts, ts)
        dsm_loss = jnp.array(0.0)

    logp_xt_t = logp_xts_ts[:BS]
    logp_xs_s = logp_xts_ts[BS:]

    logp_xs_t = log_q_fn(xs, features, t)
    logp_xt_s = log_q_fn(xt, features, s)

    Z1 = logp_xs_t - logp_xs_s
    Z2 = logp_xt_s - logp_xt_t

    per_sample = jax.nn.softplus(Z1) + jax.nn.softplus(Z2)
    clf_loss = per_sample.mean()
    return weight * clf_loss, dsm_loss


def diffclf_loss_fn(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable,
    score_fn: Callable,
    log_q_fn: Callable,
    log_q_and_score_fn: Callable,
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    *,
    k: int = 2,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if k == 2:
        return diffclf_loss_binary(
            key=key,
            sde=sde,
            dsm_loss_fn=dsm_loss_fn,
            score_fn=score_fn,
            log_q_fn=log_q_fn,
            log_q_and_score_fn=log_q_and_score_fn,
            x=x,
            features=features,
            t=t,
            weight=weight,
            eps=eps,
        )
    elif k > 2:
        return diffclf_loss_multi(
            key=key,
            sde=sde,
            dsm_loss_fn=dsm_loss_fn,
            score_fn=score_fn,
            log_q_fn=log_q_fn,
            log_q_and_score_fn=log_q_and_score_fn,
            x=x,
            features=features,
            t=t,
            weight=weight,
            eps=eps,
            k=k,
        )
    else:
        raise ValueError(f"Invalid k: {k}")