from PIL import Image
import os

from sklearn import linear_model
from sklearn.externals import joblib


def getImageData(filename):
	x = Image.open(filename)
	dat = list(x.getdata())
	return dat

# For Unix-based systems only. Outputs the filenames of the data directory into a text file to be read
os.system('ls ../Competition/data/ > files.txt')

# Places all filenames in files.txt into a list. Alternatively, replace the previous line and following block with:
# fnames = os.listdir('../Competition/data/')
f = open('files.txt', 'r')
fnames = []
s = f.readline()
while(s is not ''):
	fnames.append(str(s)[:-1])
	s = f.readline()
f.close()


#Creates a dictionary, dataMap, and places the image data into it. Each character A-Z is a key, corresponding to a list of the image datas for that letter
dataMap = {}
for fil in fnames:
	if(not fil[0] in dataMap.keys()):
		dataMap[fil[0]]=[]
	dataMap[fil[0]].append(getImageData('../Competition/data/'+fil))


#obtains keys
ks = list(dataMap.keys())
ks.sort()
it = 0

#Creates logistic regression model with no additional paramters
#Check sci kit learn documentation for optional tuning parameters
LR = linear_model.LogisticRegression()

#Places all data into one list, and all labels into another for training
#it variable is used to assign an integer value to each key, or character. A = 0, B = 1,...
totalData = []
totalTarget = []
for k in ks:
	d = dataMap[k]
	tot = len(d)
	for i in range(0, tot):
		totalTarget.append(it)

	totalData+=d

	it+=1

#Trains the model with the training data
LR.fit(totalData, totalTarget)

#Writes out the logistic regression model to a set of .pkl files
joblib.dump(LR, 'logreg.pkl')