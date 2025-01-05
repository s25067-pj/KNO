import sys

import tensorflow as tf


def set_of_equations_resolver(A, B):

    A_tensor = tf.constant(A, dtype = tf.float32)
    B_tensor = tf.constant(B, dtype=tf.float32)

    A_tensor_rotate = tf.linalg.inv(A_tensor)

    solve_problem = tf.matmul(A_tensor_rotate, B_tensor)

    return solve_problem

if __name__ == "__main__":

    n = int(sys.argv[1])
    A = []
    B = []

    for i in range(n):
        a = []
        for c in range(n):
            c = float(sys.argv[c*n+i+2])
            a.append(c)
        A.append(a)

    for i in range(n):
        b = []
        d = float(sys.argv[n*n+i+2])
        b.append(d)
        B.append(b)

    print(A)
    print(B)

    check_A = tf.linalg.det(A)
    print(check_A)

    if(check_A == 0):
        print(f'Złe dane')
    else:
        set_of_equations_resolver = set_of_equations_resolver(A, B)

        print(f'Wynik ukladu rownan: {set_of_equations_resolver}')


