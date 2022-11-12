import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import tensorflow as tf
import t_SNE
from jax import random
import jax.numpy as jnp
import logging
import matplotlib.pyplot as plt
import optax


log = logging.getLogger(__name__)


color = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
    3: (1.0, 1.0, 0.0),
    4: (1.0, 0.0, 1.0),
    5: (0.0, 1.0, 1.0),
    6: (0.3, 0.3, 0.7),
    7: (0.3, 0.7, 0.3),
    8: (0.7, 0.3, 0.3),
    9: (0.7, 0.7, 0.3),
}


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / (255. / 2) - 1.
    return tf.reshape(images, (28 * 28,)), labels


def get_data(n=1024, filter_label=None):
    dataset, = tfds.load(
        'mnist',
        split=['train'],
        as_supervised=True,
        with_info=False,
    )
    if filter_label is not None:
        dataset = dataset.filter(lambda _, label: label == filter_label)
    data = list(dataset
        .map(preprocess)
        .take(n)
        .as_numpy_iterator()
    )
    images = jnp.array(tuple(a for a, b in data))
    labels = jnp.array(tuple(b for a, b in data))
    return images, labels


def full_MNIST():
    n = 512
    desired_perplexity = 30.0
    reduction_dimension = 3
    learning_rate = 1.0

    key = random.PRNGKey(0)
    data, labels = get_data(n)
    optimizer = optax.adam(learning_rate)
    dummy = jnp.zeros((n, reduction_dimension))
    learner_state = optimizer.init(dummy)

    def update_func(data, grad):
        nonlocal learner_state
        updates, learner_state = optimizer.update(grad, learner_state)
        return optax.apply_updates(data, updates)

    def termination_func(step, value, grad):
        return step >= 1000 or jnp.max(jnp.abs(grad)) < 1e-4

    reduced = t_SNE.t_SNE(
        key,
        data,
        desired_perplexity,
        reduction_dimension,
        update_func,
        termination_func
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*reduced.T, c=[color[int(l)] for l in labels])
    plt.show()
    plt.close(fig)


def MNIST_by_digit():
    n = 128
    desired_perplexity = 10.0
    reduction_dimension = 2
    learning_rate = 1.0

    key = random.PRNGKey(0)
    data = jnp.stack(tuple(get_data(n, filter_label=i)[0] for i in range(1, 10)), axis=0)

    optimizer = optax.adam(learning_rate)
    dummy = jnp.zeros((9, n, reduction_dimension))
    learner_state = optimizer.init(dummy)

    def update_func(data, grad):
        nonlocal learner_state
        updates, learner_state = optimizer.update(grad, learner_state)
        return optax.apply_updates(data, updates)

    def termination_func(step, value, grad):
        return step >= 1000 or jnp.max(jnp.abs(grad)) < 1e-5

    reduced = t_SNE.t_SNE(
        key,
        data,
        desired_perplexity,
        reduction_dimension,
        update_func,
        termination_func
    )

    fig = plt.figure()
    for i in range(1, 10):
        ax = fig.add_subplot(3, 3, i)
        ax.scatter(*reduced[i - 1].T)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # forbid nans
    from jax.config import config
    config.update("jax_debug_nans", True)

    # configure logging
    root_logger = logging.getLogger()
    formatter = logging.Formatter('{relativeCreated:12.3f}   {levelname: <8}  {name: <15} {message}', style='{')
    root_handler = root_logger.handlers[0].setFormatter(formatter)
    root_logger.setLevel(logging.DEBUG)
    for name in ['matplotlib', 'absl']:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    #
    # MNIST_by_digit()
    full_MNIST()