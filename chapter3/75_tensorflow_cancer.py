import tensorflow as tf
import numpy as np
import pandas as pd
train=pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')
test=pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')

X_train=np.float32(train[['Clump Thickness','Cell Size']].T)
y_train=np.float32(train['Type'].T)
X_test=np.float32(test['Type'].T)
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,X_train)+b

loss=tf.reduce_mean(tf.square(y-y_train))
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for step in range(0,1000):
    sess.run(train)
    if step%200==0:
        print(step,sess.run(W),sess.run(b))