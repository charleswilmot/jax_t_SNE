import jax
import jax.numpy as jnp
from jax import random
import logging
from functools import partial
import numpy as np
from itertools import count
from typing import Callable


log = logging.getLogger(__name__)


def _conditional_probabilities(
        squared_distance_matrix: jax.Array,
        desired_perplexity: float) -> jax.Array:
    """
    Performs a binary search to find the values betas, such that the entropy of
    the distribution matches the desired entropy `H = log(perplexity)`. Returns
    the gaussian conditional probabilities with the tuned betas.

    :param squared_distance_matrix: array with shape [..., N, N] such that
    `squared_distance_matrix[..., i, j]` is the squared euclidian distance
    between input datapoint `i` and input datapoint `j`
    :type squared_distance_matrix: class:`jnp.array`

    :param desired_perplexity: exponential of the desired entropy of the
    distribution
    :type desired_perplexity: class: `float`
    """
    N_STEPS = 100
    zero_diag = (1 - jnp.identity(squared_distance_matrix.shape[-2]))
    desired_entropy = jnp.log(desired_perplexity)
    mean_abs = jnp.mean(
        jnp.abs(squared_distance_matrix),
        axis=-1,
        keepdims=True
    )
    betas = 1 / mean_abs  # initial guess for beta
    shape = squared_distance_matrix.shape[:-1] + (1,)
    betas_max = jnp.full(shape=shape, fill_value=+jnp.inf)
    betas_min = jnp.full(shape=shape, fill_value=-jnp.inf)

    for _ in range(N_STEPS):
        P = jnp.exp(-squared_distance_matrix * betas)             # [..., N, N]
        P *= zero_diag                                            # [..., N, 1]
        sum_P = jnp.sum(P, axis=-1, keepdims=True)                # [..., N, N]
        P /= sum_P
        sum_disti_P = jnp.sum(
            squared_distance_matrix * P,
            axis=-1,
            keepdims=True
        )                                                         # [..., N, 1]
        entropies = jnp.log(sum_P) + betas * sum_disti_P          # [..., N, 1]
        entropies_diff = entropies - desired_entropy              # [..., N, 1]
        predicate = entropies_diff > 0                            # [..., N, 1]
        betas_max = jnp.where(predicate, betas_max, betas)        # [..., N, 1]
        betas_min = jnp.where(predicate, betas, betas_min)        # [..., N, 1]
        betas_inc = jnp.where(
            betas_max == jnp.inf,
            betas * 2,
            (betas + betas_max) / 2
        )                                                         # [..., N, 1]
        betas_dec = jnp.where(
            betas_min == -jnp.inf,
            betas / 2,
            (betas + betas_min) / 2,
        )                                                         # [..., N, 1]
        betas = jnp.where(predicate, betas_inc, betas_dec)        # [..., N, 1]
    return P


def _input_similarity(
        squared_distance_matrix: jax.Array,
        desired_perplexity: float) -> jax.Array:
    """
    Get the input data's similarity matrix with a given perplexity

    :param squared_distance_matrix: array with shape [..., N, N] such that
    `squared_distance_matrix[..., i, j]` is the squared euclidian distance
    between input datapoint `i` and input datapoint `j`
    :type squared_distance_matrix: class: `jnp.array`

    :param desired_perplexity: exponential of the desired entropy of the
    distribution
    :type desired_perplexity: class: `float`
    """
    conditional_P = _conditional_probabilities(
        squared_distance_matrix, desired_perplexity)
    P = conditional_P + jnp.swapaxes(conditional_P, -2, -1)
    sum_P = jnp.sum(P, axis=(-2, -1), keepdims=True)
    P = P / sum_P
    return P


def _output_similarity(
        squared_distance_matrix: jax.Array,
        degrees_of_freedom: int) -> jax.Array:
    """
    Get the output data's similarity matrix with a given perplexity

    :param squared_distance_matrix: array with shape [..., N, N] such that
    `squared_distance_matrix[..., i, j]` is the squared euclidian distance
    between output datapoint `i` and output datapoint `j`
    :type squared_distance_matrix: class: `jnp.array`

    :param degrees_of_freedom: Degrees of freedom of the Student's
    t-distribution. See "Learning a Parametric Embedding by Preserving
    Local Structure" by Laurens van der Maaten, 2009, for a good value for
    this parameter
    :type degrees_of_freedom: class: `int`
    """
    squared_distance_matrix /= degrees_of_freedom
    squared_distance_matrix += 1
    squared_distance_matrix **= (degrees_of_freedom + 1.0) / -2.0
    Q = squared_distance_matrix / 2 / \
        jnp.sum(squared_distance_matrix, axis=(-2, -1))[..., None, None]
    return Q


def squared_distance_matrix(data: jax.Array) -> jax.Array:
    """
    Computes the full distance matrix between the samples of a dataset. The
    last dimension is interpreted as the feature dimension, the
    one-beofore-last dimension is interpreted as the dataset size dimension.
    All others are interpreted as the batch dimensions.

    :param data: array with shape [..., N, D]
    """
    return jnp.sum((
        data[..., None, :, :] -                       # [..., 1, N, D]
        data[..., None, :]                            # [..., N, 1, D]
    ) ** 2, axis=-1)                                  # [..., N, N]


def _kl_divergences(P: jax.Array, Q: jax.Array) -> jax.Array:
    """
    Computes the KL-Divergence between P and Q.
    """
    indices_a, indices_b = jnp.triu_indices(P.shape[-2], k=1)
    P_sup = P[..., indices_a, indices_b]
    Q_sup = Q[..., indices_a, indices_b]
    return 2 * jnp.sum(P_sup * jnp.log(P_sup / Q_sup), axis=-1)


@jax.jit
def _t_SNE_loss(output_data: jax.Array, P: jax.Array):
    """
    Given the output data (over which the optimization must be performed) and
    the input data's similarity matrix, returns the KL-Divergence between the
    input data's similarity matrix and the output data's similarity matrix.
    Optimizing the values in the array output_data so as to minimize the output
    of this function yields a repartition of the samples in output_data the
    resulting output similarity matrix resembles the input similarity matrix.

    :param output_data: dataset of samples in the reduced dimensionality
    :type output_data: class `jnp.array`
    :param P: the input data (ie high dimensionality) 's similarity matrix
    :type P: class `jnp.array`
    """
    reduction_dimension = output_data.shape[-1]
    output_squared_distance_matrix = squared_distance_matrix(
        output_data)  # [..., N, N]
    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = reduction_dimension - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(reduction_dimension - 1, 1)
    Q = _output_similarity(output_squared_distance_matrix, degrees_of_freedom)
    kl_divergences = _kl_divergences(P, Q)                        # [..., N, N]
    return jnp.sum(kl_divergences)


_t_SNE_loss_value_grad = jax.value_and_grad(_t_SNE_loss)


def t_SNE(key: random.PRNGKey,
        input_data: jax.Array,
        desired_perplexity: float,
        reduction_dimension: int,
        update_func: Callable[[jax.Array, jax.Array], jax.Array],
        termination_func: Callable[[int, jax.Array, jax.Array], bool],
        z_score: bool=False):
    """
    Performs t-SNE dimensionality reduction using the jax library.

    :param key: a PRNGKey to initialize the output datasamples normally
    :type key: class: `jax.random.PRNGKey`
    :param input_data: the data to be reduced, with shape [..., N, D_IN] where
    N is the dtaset size and D_IN is the feature size. All other dimensions are
    interpreted as batch dimensions.
    :type input_data: class: `jnp.array`
    :param desired_perplexity: exponential of the desired entropy of the
    distribution
    :type desired_perplexity: class: `float`
    :param reduction_dimension: the dimension D_OUT of the reduced data.
    :type reduction_dimension: class: `int`
    :param update_func: a callable object that takes as input the reduced
    output_data and the gradiend of the loss with respect to the output_data,
    and returns the new value for the output_data.
    :type update_func: callable
    :param termination_func: a callable object called at every iteration of the
    loop. The loop ends when this callable returns True. The callable takes 3
    arguments: step, the number of iteration passed, value, the value of the KL-
    divergence at this iteration, and grad, the value of the gradient of the KL-
    divergence with respect to the output_data.
    :type termination_func: callable
    :param z_score: whether or not the data should be z_scored before being
    reduced. Defaults to False
    :type z_score: class: `bool`
    """
    if z_score:
        mean = jnp.mean(input_data, axis=-1, keepdims=True)   # [..., 1, D_IN]
        std = jnp.std(input_data, axis=-1, keepdims=True)     # [..., 1, D_IN]
        input_data = (input_data - mean) / std                # [..., N, D_IN]
    output_shape = input_data.shape[:-1] + (reduction_dimension,)
    output_data = random.normal(key, shape=output_shape)      # [..., N, D_OUT]
    short = squared_distance_matrix(input_data)               # [..., N, N]
    input_squared_distance_matrix = short
    P = _input_similarity(input_squared_distance_matrix, desired_perplexity)

    for step in count():
        value, grad = _t_SNE_loss_value_grad(output_data, P)
        if termination_func(step, value, grad):
            break
        output_data = update_func(output_data, grad)
        log.debug(f'value = {np.array(value)}')
    return output_data


if __name__ == '__main__':
    root_logger = logging.getLogger()
    fmt = '{relativeCreated:12.3f}   {levelname: <8}  {name: <15} {message}'
    formatter = logging.Formatter(fmt, style='{')
    root_handler = root_logger.handlers[0].setFormatter(formatter)
    root_logger.setLevel(logging.DEBUG)
    for name in ['matplotlib', 'absl']:
        logging.getLogger(name).setLevel(logging.WARNING)

    key = random.PRNGKey(0)
    data = random.uniform(key, shape=(1000, 10))
    reduced = t_SNE(key, data, 100.0, 3)
    log.debug(f'{reduced}')
