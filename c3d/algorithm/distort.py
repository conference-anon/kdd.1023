import numpy as np
import tensorflow as tf


def random_attractors(pts, n_attractors=6):
    pts = pts[np.all(~np.isnan(pts), axis=1)]
    mean = np.mean(pts, axis=0)
    avg_radius = np.mean(np.linalg.norm(pts - mean, axis=1))
    radius = avg_radius / 2
    n_pts = pts.shape[0]

    indices = np.random.choice(n_pts, n_attractors)
    base = pts[indices]
    return base + np.random.uniform(low=-radius, high=+radius, size=(n_attractors, pts.shape[1]))


def tf_random_attractors(pts, n_attractors):
    # batch_size, point, features
    assert len(pts.shape) == 3
    with tf.name_scope('random_attractors'):
        mean = tf.reduce_mean(pts, axis=-1, keepdims=True)
        avg_radius = tf.reduce_mean(tf.linalg.norm(pts - mean, axis=-1), axis=-1)
        # [batch]
        radius = avg_radius / 2

        shape = tf.shape(pts)
        batch_size = shape[0]
        n_pts = shape[1]

        base_indices = tf.random_uniform(
            shape=[batch_size, n_attractors], minval=0, maxval=n_pts,
            dtype=tf.int32
        )
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        batch_indices = tf.expand_dims(batch_indices, -1)
        batch_indices = tf.tile(batch_indices, (1, n_attractors))

        # [batch, n_attractors, 2]
        base = tf.gather_nd(pts, tf.stack([batch_indices, base_indices], axis=-1))
        rad_expand = tf.expand_dims(tf.expand_dims(radius, axis=-1), axis=-1)
        return base + tf.random_uniform(
            minval=-rad_expand, maxval=rad_expand,
            shape=tf.shape(base)
        )


def attract_pts(pts, attractors, beta=8, gamma=1):
    n_pts = pts.shape[0]
    n_attractors = attractors.shape[0]

    # [n_pts, n_attractors]
    distances = np.linalg.norm(
        np.expand_dims(pts, 1) - np.expand_dims(attractors, 0),
        axis=-1
    )

    # Normalize all distances to a mean of 1, to make sure the noising
    # function doesn't vary with the scale of the input
    distances /= np.nanmean(distances)

    # [n_pts, n_attractors]
    weights = np.exp(- distances * beta)

    # The neutral weight should be 10 times the average sum of weights
    # from all attractors.
    neutral = gamma * np.nanmean(np.sum(weights, 1))

    # [n_pts, 1 + n_attractors]
    weights = np.concatenate((weights, neutral * np.ones((n_pts, 1))), axis=1)
    weight_sums = np.sum(weights, axis=1, keepdims=True)

    # [n_pts, 1 + n_attractors, 1]
    weights = np.expand_dims(weights, axis=-1)

    # [n_pts, 1 + n_attractors, dim]
    candidates = np.concatenate(
        (np.tile(np.expand_dims(attractors, axis=0),
                 (n_pts, 1, 1)),
         np.expand_dims(pts, axis=1)
         ), axis=1
    )

    return np.sum(weights * candidates, axis=1) / weight_sums


def tf_attract_pts(pts, attractors, beta=8, gamma=1):
    n_pts = tf.shape(pts)[1]
    n_attractors = tf.shape(pts)[1]

    # [batch, n_pts, n_attractors]
    distances = tf.linalg.norm(
        tf.expand_dims(pts, -2) - tf.expand_dims(attractors, -3),
        axis=-1
    )

    # Normalize all distances to a mean of 1, to make sure the noising
    # function doesn't vary with the scale of the input
    distances /= tf.reduce_mean(
        tf.reduce_mean(distances, axis=-1, keepdims=True),
        axis=-2, keepdims=True
    )

    # [batch, n_pts, n_attractors]
    weights = tf.exp(- distances ** 3 * beta)

    # The neutral weight should be 10 times the average sum of weights
    # from all attractors.
    # [batch]
    neutral = gamma * tf.reduce_mean(tf.reduce_sum(weights, axis=-1), axis=-1)

    # [batch, n_pts, 1 + n_attractors]
    weights = tf.concat(
        (weights,
         tf.tile(
             tf.expand_dims(tf.expand_dims(neutral, axis=-1), axis=-1),
             (1, n_pts, 1)
         )
         ),
        axis=2
    )
    # [batch, n_pts, 1]
    weight_sums = tf.reduce_sum(weights, axis=-1, keepdims=True)

    # [batch, n_pts, 1 + n_attractors, 1]
    weights = tf.expand_dims(weights, axis=-1)

    # [batch, n_pts, 1 + n_attractors, dim]
    candidates = tf.concat(
        (tf.tile(tf.expand_dims(attractors, axis=1),
                 (1, n_pts, 1, 1)),
         tf.expand_dims(pts, axis=2)
         ), axis=2
    )

    return tf.reduce_sum(weights * candidates, axis=2) / weight_sums


def random_attract_pts(pts, n_attractors=6, **kwargs):
    attractors = random_attractors(pts, n_attractors=n_attractors)
    return attract_pts(pts, attractors, **kwargs)


def tf_random_attract_pts(pts, n_attractors):
    attractors = tf_random_attractors(pts, n_attractors)
    return tf_attract_pts(pts, attractors)
