from PIL import Image
import os
import sys

from sklearn import linear_model
from sklearn.externals import joblib

def getImageData(filename):
	x = Image.open(filename)
	dat = list(x.getdata())
	return dat

#Obtains test directory as argumet. Adds / if there is none
testdir = str(sys.argv[1])
if(testdir[-1]!='/'):
	testdir+='/'

#Obtains all filenames in the given directory
fnames = os.listdir(testdir)

#Obtains testing data for all images in the given testing directory
tdata = []
for n in fnames:
	tdata.append(getImageData(testdir)+n))

#loads logistic regression model trained in challenge1.py
LR = joblib.load('logreg.pkl')

#Obtains predicted results from the model
results = LR.predict(tdata)

#converts integer predicted results to chracters
chresults = []
for i in range(0, len(results)):
	chresults.append(chr(int(results[i])+65))

#Obtains the ground truth as the first letter of each filename in the testing directory
target = []
for i in range(0, len(fnames)):
	target.append(fnames[i][0])

#Prints out error based on ground truth
mistakes = 0
for i in range(0, len(chresults)):
	if(chresults[i]!=target[i]):
		mistakes+=1
		#print(fnames[i]+'     '+chresults[i])
print(mistakes/len(chresults))


#Outputs results to a csv
f2 = open('output.csv', 'w')
for i in range(0, len(fnames)):
	f2.write(fnames[i]+','+chresults[i]+'\n')
f2.close()
