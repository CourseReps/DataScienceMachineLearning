__Author__ = "Austin Taghavi"

from PIL import Image
import os
import string
import random
import sys

def getDistance(pixels1, pixels2):
	dist = 0.0
	for i in range(len(pixels1)):
		dist += (pixels1[i] - pixels2[i])**2
	return dist


def KNearestNeighbors(directoryName, K=3):
	testfilenames = os.listdir(directoryName)
	num = 0
	errors = 0
	K = 5

	for filename in testfilenames:
			im = Image.open(directoryName + "/" + filename)
			pixels = list(im.getdata())
			for i in range(len(pixels)):
				pixels[i] = pixels[i]*1.0/255.0
			minDist = 99999.0
			distances = []
			for elem in data:
				distance = getDistance(pixels, elem[0])
				tup = (distance, elem[1])
				distances.append(tup)
			distances.sort(key=lambda tup: tup[0])
			Knearest = distances[:K]
			cnts = {}
			for elem in Knearest:
				if elem[1] not in cnts:
					cnts[elem[1]] = 1
				else:
					cnts[elem[1]] += 1
			expectedClass = 'A'
			maxx = 0
			for key in cnts:
				if cnts[key] > maxx:
					maxx = cnts[key]
					expectedClass = key
			num += 1
			if expectedClass is not filename[:1]:
				errors += 1
			print filename + "," + expectedClass
	#print errors*1.0/num*1.0

filenames = os.listdir("../Competition/data")
data = []

testDir = sys.argv[1]
print testDir
train_names = []

for filename in filenames:
	if random.random() < 1.0:
		currentLetter = filename[:1]
		im = Image.open("../Competition/data/" + filename)
		pixels = list(im.getdata())
		for i in range(len(pixels)):
			pixels[i] = pixels[i]*1.0/255.0
		tup = (pixels, currentLetter)
		data.append(tup)
		train_names.append(filename)


KNearestNeighbors(testDir, 1)

