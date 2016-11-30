import numpy as np
import argparse
from sklearn import svm, preprocessing
from PIL import Image
from os import listdir
from multiprocessing import Pool
from functools import partial
import csv


C_range = np.logspace(-2, 1, 4)
gamma_range = np.logspace(-3, 0, 4)

def inputData(pathName):
    """

    :return: Image sets in greyscale matricies from data folder
    """
    files = listdir(pathName)
    X = []
    y = []
    for file in files:
        key = file[0][0] #first letter of file name
        image = Image.open(pathName+'/'+file)
        x = np.asarray(image.convert('L')) #Converts To greyscale
        X.append(x)
        y.append(key)
    X = np.array(X)
    y = np.array(y)

    nsamples, nx, ny = X.shape
    

    return files,X,y

def getValidationAndTestSet(X,y):
    """

    :param X: Entire Data Set
    :param y: Etire target vector
    :return: [X,y,X_cv,y_cv,X_test,y_test] Training sets, cross validation sets, and test sets with the training and cv sets scaled
    """
    rnd = np.random.rand(len(X))
    cvIdx = rnd < .2
    testIdx = rnd > .8

    X_cv = X[cvIdx]
    y_cv = y[cvIdx]
    X_test = X[testIdx]
    y_test = y[testIdx]
    X = X[~cvIdx&~testIdx]
    y = y[~cvIdx&~testIdx]

    nsamples, nx, ny = X.shape
    X = preprocessing.scale(X.reshape((nsamples,nx*ny)).astype(float,casting='unsafe'))
    nsamples, nx, ny = X_test.shape
    X_test = preprocessing.scale(X_test.reshape((nsamples,nx*ny)).astype(float,casting='unsafe'))
    nsamples, nx, ny = X_cv.shape
    X_cv = preprocessing.scale(X_cv.reshape((nsamples,nx*ny)).astype(float,casting='unsafe'))


    return [X,y,X_cv,y_cv,X_test,y_test]

def trainClassifier(X,y,kernel,gamma,C):
    """

    :param X: Training set
    :param y: Training vector
    :param C: Regularization Constant
    :param gamma: Rbf constant
    :return: Trained classifiers
    """
    clf = svm.SVC(kernel=kernel,C=C,gamma=gamma)
    clf.fit(X,y)
    print("Classifier Trained for C = {}".format(C))
    return clf

def getClassifiers(X,y,kernel,gamma_range,C_range):
    """

    :param X: training set
    :param y: training vector
    :param C_range: range of C values for regularization
    :param gamma_range:  range of gamma values for rbf kernel
    :return: a list of trained classifiers
    """
    classifiers = []
    pool = Pool()

    if kernel == "rbf":
        for gamma in gamma_range:
            print("Gamma = {}".format(gamma))
            func = partial(trainClassifier,X,y,'rbf',gamma)
            classifiers = classifiers + pool.map(func,C_range)
    else:
        kernel = "linear"
        gamma = 1/X.shape[1]
        func = partial(trainClassifier,X,y,kernel,gamma)
        classifiers = classifiers + pool.map(func,C_range)

    pool.close()
    pool.join()
    return classifiers


def getBestClassifier(classifiers,X_cv,y_cv):
    """

    :param classifiers: List of trained classifiers
    :param X_cv: cross validation set
    :param y_cv: cross validation vector
    :return: the best classifier from the list according to the cv set
    """
    minError = 1
    minClf = []
    print("Getting Best Error")
    pool = Pool()
    func = partial(getErrorRate,X_cv,y_cv)
    errors = pool.map(func,classifiers)
    pool.close()
    pool.join()

    return classifiers[errors.index(min(errors))]


def getErrorRate(X_test,y_test,classifier):
    """

    :param X_test: Test set
    :param y_test: Test vector
    :param classifier: Trained classifier
    :return: Error rate for the classifier according to the test sets
    """
    prediction = classifier.predict(X_test)
    j = 0
    for i in range(0, len(prediction)):
        if prediction[i] !=  y_test[i]:
            j = j+1
    return j/len(prediction)
    



def outputCSV(dataTestPath,classifier):
    
    filenames,X_test,y_test = inputData(dataTestFolder)
    nsamples, nx, ny = X_test.shape
    X_test = preprocessing.scale(X_test.reshape((nsamples,nx*ny)).astype(float,casting='unsafe'))
    
    prediction = classifier.predict(X_test)
    
    with open('output.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['filename'] + ['predicted label'])
        #    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
    
        for i in range(0,len(y_test)):
            spamwriter.writerow([str(filenames[i])]+[y_test[i]])
            	#print test_filenames[i],char(op_labels[i])
                
    epsilon = getErrorRate(X_test,y_test,classifier)
    
    print("Successful classification rate with SVM is {}%".format(int(100-epsilon*100)))

    
    
    

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str,
                        help="Folder the training images")
    parser.add_argument("test_path", type=str,
                        help="Folder of the testing images")
    parser.add_argument("kernel", type=str,
                        help="Type of kernel to use in SVM. 'rbf' and 'linear' supported. ")
    args = parser.parse_args()
    dataFolder=args.train_path
    dataTestFolder=args.test_path
    kernel=args.kernel
    
    return dataFolder,dataTestFolder,kernel

if __name__ == '__main__':

    
    dataFolder, dataTestFolder, kernel = getArgs()
    files,X,y = inputData(dataFolder)


    print("Data Input")
    [X,y,X_cv,y_cv,X_t,y_t] = getValidationAndTestSet(X,y)
    print("Sets Made")
    classifiers = getClassifiers(X,y,kernel,gamma_range,C_range)
    print("All Classifiers Trained")
    minClf = getBestClassifier(classifiers,X_cv,y_cv)
    outputCSV(dataTestFolder,minClf)

    

