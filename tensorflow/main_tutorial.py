## Softmax regression (multi-class regression) for Digits classification
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#Couple of placeholders- We will input a value to these placeholders when we run the tf graph
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Variables- usually variables we intend to train in our Learning algorithm
#<x[1,:],W[:,j]> a cost function  for sample 1 associated with category j
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)

with sess:
	for i in range(1000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		if i%20==0:
			print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
	batch = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})
	if i%20==0:
		print(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


