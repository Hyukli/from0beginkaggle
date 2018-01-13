import tensorflow as tf
matrixl=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
product=tf.matmul(matrixl,matrix2)
linear=tf.add(product,tf.constant(2.0))
with tf.Session() as sess:
    result=sess.run(linear)
    print(result)