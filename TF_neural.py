import tensorflow as tf
import numpy as np

X = tf.Variable([[0.4, 0.2, 0.4]])
W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])
b = tf.Variable([[0.1, 0.2]])

XWb = tf.matmul(X, W) + b
y = tf.nn.relu(tf.matmul(X, W) + b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')
    print(sess.run(XWb))
    print('y:')
    print(sess.run(y))

X = tf.Variable([[0.4, 0.2, 0.4]])
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))

y = tf.nn.relu(tf.matmul(X, W) + b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('b:')
    print(sess.run(b))
    print('W:')
    print(sess.run(W))
    print('y:')
    print(sess.run(y))


def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b

    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)

    return outputs


X = tf.placeholder('float', [None, 4])
h = layer(3, 4, X, tf.nn.relu)
y = layer(2, 3, h)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (layer_X, layer_h, layer_y) = sess.run((X, h, y), feed_dict={X: X_array})
    print('input layer X:')
    print(layer_X)
    print('hidden layer h:')
    print(layer_h)
    print('output layer y:')
    print(layer_y)


