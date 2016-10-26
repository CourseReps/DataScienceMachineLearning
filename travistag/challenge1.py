from PIL import Image
import os

from sklearn import linear_model
from sklearn.externals import joblib


def getImageData(filename):
	x = Image.open(filename)
	dat = list(x.getdata())
	return dat


os.system('ls ../Competition/data/ > files.txt')

f = open('files.txt', 'r')
fnames = []
s = f.readline()
while(s is not ''):
	fnames.append(str(s)[:-1])
	s = f.readline()
f.close()


dataMap = {}
for fil in fnames:
	if(not fil[0] in dataMap.keys()):
		dataMap[fil[0]]=[]
	dataMap[fil[0]].append(getImageData('../Competition/data/'+fil))

ks = list(dataMap.keys())
ks.sort()
it = 0

LR = linear_model.LogisticRegression()

totalData = []
totalTarget = []
for k in ks:
	d = dataMap[k]
	tot = len(d)
	for i in range(0, tot):
		totalTarget.append(it)

	totalData+=d

	it+=1

print(totalTarget)
LR.fit(totalData, totalTarget)

joblib.dump(LR, 'logreg.pkl')