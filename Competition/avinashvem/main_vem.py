import tensorflow as tf
import numpy as np
from numpy import array
import sys
import time
import argparse
import Utils
NUM_FEATURES=2048
NUM_CLASSES=3

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str,
                    help="Path to the training images")
parser.add_argument("test_path", type=str,
                    help="Path to the testing images")
args = parser.parse_args()
path1=args.train_path	
path2=args.test_path

data_path=path1
[X,Y]=Utils.read_data(data_path,784,NUM_CLASSES)

Y_vec=np.zeros((Y.size,NUM_CLASSES))
for i in range(0,Y.size):
	Y_vec[i,Y[i]-ord('A')]=1


data_path=path2
[X_test,Y_test]=Utils.read_data(data_path,784,NUM_CLASSES)

Y_test_vec=np.zeros((Y_test.size,NUM_CLASSES))
for i in range(0,Y_test.size):
	Y_test_vec[i,Y_test[i]-ord('A')]=1
	
# tensorflow nn computational graph construction
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(tf.cast(x,tf.float32), [-1,28,28,1])

W_conv1 = Utils.weight_variable([5, 5, 1, 32])
b_conv1 = Utils.bias_variable([32])

h_conv1 = tf.nn.relu(Utils.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = Utils.max_pool_2x2(h_conv1)

W_conv2 = Utils.weight_variable([5, 5, 32, 64])
b_conv2 = Utils.bias_variable([64])

h_conv2 = tf.nn.relu(Utils.conv2d(h_pool1, W_conv2) + b_conv2) #Size 7 x 7 x64
h_pool2 = Utils.max_pool_2x2(h_conv2) #Size 7 x 7 x64

W_fc1 = Utils.weight_variable([7 * 7 * 64, NUM_FEATURES])
b_fc1 = Utils.bias_variable([NUM_FEATURES])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = Utils.weight_variable([NUM_FEATURES, NUM_CLASSES])
b_fc2 = Utils.bias_variable([NUM_CLASSES])

W_fc2 = Utils.weight_variable([NUM_FEATURES, NUM_CLASSES])
b_fc2 = Utils.bias_variable([NUM_CLASSES])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
outputs=tf.argmax(y_conv,1)

entropy_terms=tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
cross_entropy = tf.reduce_mean(entropy_terms)
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
correct_prediction = tf.cast(tf.equal(outputs, tf.argmax(y_,1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction,reduction_indices=0)

sess=tf.Session()
sess.run(tf.initialize_all_variables())
with sess:
	train_step.run(feed_dict={x: X, y_: Y_vec, keep_prob: 0.5})
	#print sess.run(y_conv[100:110,:],feed_dict={x: X, y_: Y_vec, keep_prob: 1.0})
	#print sess.run(entropy_terms,feed_dict={x: X, y_: Y_vec, keep_prob: 1.0})
	print sess.run(correct_prediction,feed_dict={x: X, y_: Y_vec, keep_prob: 1.0})
	#print(sess.run(W_fc2[:,25],feed_dict={x: X, y_: Y_vec, keep_prob: 1.0}))
	#cr_entropy=cross_entropy.eval(feed_dict={x:X, y_: Y_vec, keep_prob: 1.0})
	#print ("cross entropy %d"%cr_entropy)
	train_accuracy = sess.run(accuracy,feed_dict={x: X, y_: Y_vec, keep_prob: 1.0})
	print("training accuracy %d"%train_accuracy)

	
	#print sess.run(outputs,feed_dict={x: X_test, y_: Y_test_vec, keep_prob: 1.0})
	#test_accuracy=accuracy.eval(feed_dict={x: X_test, y_: Y_test_vec, keep_prob: 1.0})
	#print("test accuracy %g"%test_accuracy)

""""
from sklearn.neighbors import KNeighborsClassifier
print "kNN classifier with weights prop to distance"
start = time.time()
err=[]
K1=2
K2=5
for K in range(K1,K2):
    neigh = KNeighborsClassifier(n_neighbors=K,weights='distance')
    neigh.fit(X,Y)
    err.append(1-neigh.score(X_test,Y_test))
    print "K=",K, "Err=",err[K-K1]
end = time.time()
print "Elapsed time for kNN is", (end - start)


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(binarize=0.5,alpha=0.1,fit_prior=False)
clf.fit(X,Y)
print "Using Bernoulli NaiveBayes model, error is ",1-clf.score(X_test,Y_test)



from sklearn import tree
clf = tree.DecisionTreeClassifier(max_features=784,min_samples_split=10)
clf = clf.fit(X, Y)
print "Using Decision Tree classifier, error is ",1-clf.score(X_test,Y_test)
"""""