import numpy as np
from numpy import array
from PIL import Image
import os.path
import tensorflow as tf

# We will define a function that will read the .png grayscale files into a ndarray object
def read_data(data_path,vec_len,NUM_CLASSES):
    data_mat=np.empty(shape=(0,vec_len))
    labels=[]
    total_files=0
    for z in range(ord('A'),ord('A')+NUM_CLASSES):
        filenum=1
        missed_files=0
        while True:
            name1=chr(z)+str(filenum)+'.png'
            filename=data_path+name1
            if os.path.isfile(filename):
                im = Image.open(filename)
                data=np.array(im.getdata(),dtype=np.float)
                data=255.0-data
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
    #(rows,cols)=gray_scale.shape
    #for i in range(0,rows):
    # 	for j in range(0,cols):
    #		gray_scale[i][j]=round(gray_scale[i][j])
    return gray_scale, np.array(labels)
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.3, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
