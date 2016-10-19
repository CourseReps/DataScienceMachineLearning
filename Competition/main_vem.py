import pandas as pd
import numpy as np
from numpy import array
from PIL import Image
import os.path
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str,
                    help="Path to the training images")
parser.add_argument("test_path", type=str,
                    help="Path to the testing images")
args = parser.parse_args()

path1=args.train_path
path2=args.test_path
print path1, path2

# We will define a function that will read the .png grayscale files into a ndarray object
def read_data(data_path,vec_len):
    data_mat=np.empty(shape=(0,vec_len))
    labels=[]
    total_files=0
    for z in range(ord('A'),ord('Z')+1):
        filenum=1
        missed_files=0
        while True:
            name1=chr(z)+str(filenum)+'.png'
            filename=data_path+name1
            if os.path.isfile(filename):
                im = Image.open(filename)
                data=np.array(im.getdata(),dtype=np.float)
                data_mat=np.vstack((data_mat,data))
                labels.append(z)
                filenum+=1
                total_files+=1
             
            elif missed_files<20:
                missed_files+=1
                filenum+=1
            else:
                break
                
        print "Finished reading ",filenum-missed_files-1, "letter-",chr(z),"files"
    
    gray_scale=data_mat*(1/255.0)
    return gray_scale, labels


data_path=path1
[X,Y]=read_data(data_path,784)

data_path=path2
[X_test,Y_test]=read_data(data_path,784)

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
clf = tree.DecisionTreeClassifier(max_features=500,min_samples_split=10)
clf = clf.fit(X, Y)
print 1-clf.score(X_test,Y_test)
