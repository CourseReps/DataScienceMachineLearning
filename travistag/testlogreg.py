from PIL import Image
import os
import sys

from sklearn import linear_model
from sklearn.externals import joblib

def getImageData(filename):
	x = Image.open(filename)
	dat = list(x.getdata())
	return dat


os.system('ls '+str(sys.argv[1])+' > tfiles.txt')
f = open('tfiles.txt', 'r')
s = f.readline()
fnames = []
while (s is not ''):
	fnames.append(str(s)[:-1])
	s = f.readline()
f.close()
tdata = []
for n in fnames:
	tdata.append(getImageData(str(sys.argv[1])+n))

LR = joblib.load('logreg.pkl')
results = LR.predict(tdata)
chresults = []
for i in range(0, len(results)):
	chresults.append(chr(int(results[i])+65))

target = []
for i in range(0, len(fnames)):
	target.append(fnames[i][0])

mistakes = 0
for i in range(0, len(chresults)):
	if(chresults[i]!=target[i]):
		mistakes+=1
		#print(fnames[i]+'     '+chresults[i])
#print(mistakes/len(chresults))
f2 = open('output.csv', 'w')
for i in range(0, len(fnames)):
	f2.write(fnames[i]+','+chresults[i]+'\n')
f2.close()
