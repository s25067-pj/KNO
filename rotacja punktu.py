import tensorflow as tf
import math


def rotate_point(point, angle_degrees):

    angle_radians = tf.constant(math.radians(angle_degrees), dtype=tf.float64)

    rotation_matrix = tf.stack([
        [tf.cos(angle_radians), -tf.sin(angle_radians)],
        [tf.sin(angle_radians), tf.cos(angle_radians)]
    ])

    point_tensor = tf.constant(point, dtype=tf.float64)

    rotated_point = tf.linalg.matvec(rotation_matrix, point_tensor)

    return rotated_point


point = [14, -2]
angle = 181
rotated_point = rotate_point(point, angle)
print(f'Obrócony punkt: {rotated_point.numpy()}')
