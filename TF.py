import tensorflow as tf

ts_c = tf.constant(2, name='ts_c')
ts_x = tf.Variable(ts_c+5, name='ts_x')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('ts_c=', sess.run(ts_c))
    print('ts_x=', sess.run(ts_x))


width = tf.placeholder('int32', name='width')
height = tf.placeholder('int32', name='height')
area = tf.multiply(width, height, name='area')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=', sess.run(area, feed_dict={width: 6, height: 8}))

ts_X = tf.Variable([0.4, 0.2, 0.4])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X = sess.run(ts_X)
    print(X)

X = tf.Variable([[1., 1., 1.]])

W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])

XW = tf.matmul(X, W)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XW))

a = tf.Variable([0.1, 0.2, 0.3])
b = tf.Variable([0.3, 0.2, 0.1])
Sum = a + b

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(Sum))
