from PIL import Image
import os
import sys

from sklearn import linear_model
from sklearn.externals import joblib

def getImageData(filename):
	x = Image.open(filename)
	dat = list(x.getdata())
	return dat

#Simliarly to challenge1.py, obtains all filenames in the given directory. Can be replaced with a call to os.listdir(str(sys.argv[1]))
os.system('ls '+str(sys.argv[1])+' > tfiles.txt')
f = open('tfiles.txt', 'r')
s = f.readline()
fnames = []
while (s is not ''):
	fnames.append(str(s)[:-1])
	s = f.readline()
f.close()

#Obtains testing data for all images in the given testing directory
tdata = []
for n in fnames:
	tdata.append(getImageData(str(sys.argv[1])+n))

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
