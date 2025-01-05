import tensorflow as tf


def set_of_equations_resolver(A, B):

    A_tensor = tf.constant(A, dtype = tf.float32)
    B_tensor = tf.constant(B, dtype=tf.float32)

    A_tensor_rotate = tf.linalg.inv(A_tensor)

    solve_problem = tf.matmul(A_tensor_rotate, B_tensor)

    return solve_problem


A = [[2.0, -4.0],
     [-3.0, 5.0]]
B = [[5.0], [-2.0]]
set_of_equations_resolver = set_of_equations_resolver(A, B)

print(f'Wynik ukladu rownan: {set_of_equations_resolver}')

