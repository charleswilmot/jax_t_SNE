import jax
import jax.numpy as jnp
from jax import random
import logging
from functools import partial
import numpy as np
from itertools import count


log = logging.getLogger(__name__)


def _conditional_probabilities(squared_distance_matrix, desired_perplexity):
    N_STEPS = 100
    zero_diag = (1 - jnp.identity(squared_distance_matrix.shape[-2]))
    desired_entropy = jnp.log(desired_perplexity)
    betas = 1 / jnp.mean(jnp.abs(squared_distance_matrix), axis=-1, keepdims=True) # initial guess for beta
    betas_max = jnp.full(shape=squared_distance_matrix.shape[:-1] + (1,), fill_value=jnp.inf)
    betas_min = jnp.full(shape=squared_distance_matrix.shape[:-1] + (1,), fill_value=-jnp.inf)
    for _ in range(N_STEPS):
        P = jnp.exp(-squared_distance_matrix * betas)              # [..., N, N]
        P *= zero_diag
        sum_P = jnp.sum(P, axis=-1, keepdims=True)                 # [..., N, 1]
        # sum_P = jnp.clip(sum_P, EPSILON)
        P /= sum_P                                                 # [..., N, N]
        sum_disti_P = jnp.sum(
                squared_distance_matrix * P,
                axis=-1,
                keepdims=True
        )                                                          # [..., N, 1]
        entropies = jnp.log(sum_P) + betas * sum_disti_P           # [..., N, 1]
        entropies_diff = entropies - desired_entropy               # [..., N, 1]
        predicate = entropies_diff > 0                             # [..., N, 1]
        betas_max = jnp.where(
            predicate,
            betas_max,
            betas,
        )                                                          # [..., N, 1]
        betas_min = jnp.where(
            predicate,
            betas,
            betas_min,
        )                                                          # [..., N, 1]
        betas_inc = jnp.where(
            betas_max == jnp.inf,
            betas * 2,
            (betas + betas_max) / 2,
        )                                                          # [..., N, 1]
        betas_dec = jnp.where(
            betas_min == -jnp.inf,
            betas / 2,
            (betas + betas_min) / 2,
        )                                                          # [..., N, 1]
        betas = jnp.where(
            predicate,
            betas_inc,
            betas_dec,
        )                                                          # [..., N, 1]
    return P


def _input_similarity(squared_distance_matrix, desired_perplexity):
    conditional_P = _conditional_probabilities(squared_distance_matrix, desired_perplexity)
    P = conditional_P + jnp.swapaxes(conditional_P, -2, -1)
    sum_P = jnp.sum(P, axis=(-2, -1), keepdims=True)
    P = P / sum_P
    return P


def _output_similarity(squared_distance_matrix, degrees_of_freedom):
    squared_distance_matrix /= degrees_of_freedom
    squared_distance_matrix += 1
    squared_distance_matrix **= (degrees_of_freedom + 1.0) / -2.0
    Q = squared_distance_matrix / 2 / jnp.sum(squared_distance_matrix, axis=(-2, -1))[..., None, None]
    return Q


def squared_distance_matrix(data):
    return jnp.sum((
        data[..., None, :, :] -                       # [..., 1, N, D]
        data[..., None, :]                            # [..., N, 1, D]
    ) ** 2, axis=-1)                                  # [..., N, N]


def _kl_divergences(P, Q):
    indices_a, indices_b = jnp.triu_indices(P.shape[-2], k=1)
    P_sup = P[..., indices_a, indices_b]
    Q_sup = Q[..., indices_a, indices_b]
    return 2 * jnp.sum(P_sup * jnp.log(P_sup / Q_sup), axis=-1)


@jax.jit
def _t_SNE_loss(output_data, P):
    reduction_dimension = output_data.shape[-1]
    output_squared_distance_matrix = squared_distance_matrix(output_data) # [..., N, N]
    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = reduction_dimension - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(reduction_dimension - 1, 1)
    Q = _output_similarity(output_squared_distance_matrix, degrees_of_freedom)
    kl_divergences = _kl_divergences(P, Q)                              # [..., N, N]
    return jnp.sum(kl_divergences)


_t_SNE_loss_value_grad = jax.value_and_grad(_t_SNE_loss)


def t_SNE(key, input_data, desired_perplexity, reduction_dimension, update_func, termination_func, z_score=False):
    if z_score:
        mean = jnp.mean(input_data, axis=-1, keepdims=True) # [..., 1, D_IN]
        std = jnp.std(input_data, axis=-1, keepdims=True)   # [..., 1, D_IN]
        input_data = (input_data - mean) / std              # [..., N, D_IN]
    output_shape = input_data.shape[:-1] + (reduction_dimension,)
    output_data = random.normal(
        key,
        shape=output_shape,
    )                                                       # [..., N, D_OUT]
    input_squared_distance_matrix = squared_distance_matrix(input_data) # [..., N, N]
    P = _input_similarity(input_squared_distance_matrix, desired_perplexity)

    for step in count():
        value, grad = _t_SNE_loss_value_grad(output_data, P)
        if termination_func(step, value, grad): break
        output_data = update_func(output_data, grad)
        log.debug(f'value = {np.array(value)}')
    return output_data


if __name__ == '__main__':
    root_logger = logging.getLogger()
    formatter = logging.Formatter('{relativeCreated:12.3f}   {levelname: <8}  {name: <15} {message}', style='{')
    root_handler = root_logger.handlers[0].setFormatter(formatter)
    root_logger.setLevel(logging.DEBUG)
    for name in ['matplotlib', 'absl']:
        logging.getLogger(name).setLevel(logging.WARNING)

    key = random.PRNGKey(0)
    data = random.uniform(key, shape=(1000, 10))
    reduced = t_SNE(key, data, 100.0, 3)
    log.debug(f'{reduced}')
