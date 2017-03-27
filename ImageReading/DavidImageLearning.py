from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

RealImages = []
SyntheticImages = []
RealImagesRatio = []
SyntheticImagesRatio = []

NumLandscapes = 40
NumSynthetic = 40
NumReal = 89


for root, dirs, files in os.walk("CompGenLandscapes"):
	for file in files[:NumLandscapes]:
		if(file != '.DS_Store'):
			img_mp = mpimg.imread('CompGenLandscapes/'+file)
			SyntheticImages.append(img_mp)


for root, dirs, files in os.walk("SyntheticImages"):
	for file in files[:NumSynthetic]:
		if(file != '.DS_Store'):
			img_mp = np.uint8(mpimg.imread('SyntheticImages/'+file)*256)
			SyntheticImages.append(img_mp)

print("Imported all Synthetic Images")

for img in SyntheticImages:
	lum_img_red = img[:,:,0]
	lum_img_green = img[:,:,1]
	lum_img_blue = img[:,:,2]
	hist_red = np.histogram(lum_img_red, bins=256)
	hist_green = np.histogram(lum_img_green, bins=256)
	hist_blue = np.histogram(lum_img_blue, bins=256)
	SyntheticImagesRatio.append([np.max(hist_red[0])/np.mean(hist_red[0]),
								np.max(hist_green[0])/np.mean(hist_green[0]),
								np.max(hist_blue[0])/np.mean(hist_blue[0]),0])

print("Obtained Ratios for Synthetic Images")

for root, dirs, files in os.walk("RealImages"):
	for file in files[:NumReal]:
		if(file != '.DS_Store'):
			img_mp = mpimg.imread('RealImages/'+file)
			RealImages.append(img_mp)

print("Imported all Real Images")

for img in RealImages:
	lum_img_red = img[:,:,0]
	lum_img_green = img[:,:,1]
	lum_img_blue = img[:,:,2]
	hist_red = np.histogram(lum_img_red, bins=256)
	hist_green = np.histogram(lum_img_green, bins=256)
	hist_blue = np.histogram(lum_img_blue, bins=256)
	RealImagesRatio.append([np.max(hist_red[0])/np.mean(hist_red[0]),
								np.max(hist_green[0])/np.mean(hist_green[0]),
								np.max(hist_blue[0])/np.mean(hist_blue[0]),1])

print("Obtained Ratios for Real Images")

AllImagesRatio = np.concatenate((SyntheticImagesRatio,RealImagesRatio), axis=0)

train, test = train_test_split(AllImagesRatio, test_size = 0.2)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train[:,:3])
numcorrect = 0
for testimage in test:
	distance, index = nbrs.kneighbors([testimage[:3]])
	# print("Guess: ", train[index[0]][0][3])
	# print("Actual Value: ", testimage[3])
	if(train[index[0]][0][3] == testimage[3]):
		numcorrect = numcorrect + 1

print(numcorrect/len(test))

train, test = train_test_split(AllImagesRatio, test_size = 0.2)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train[:,:3])
numcorrect = 0
for testimage in test:
	distance, index = nbrs.kneighbors([testimage[:3]])
	# print("Guess: ", train[index[0]][0][3])
	# print("Actual Value: ", testimage[3])
	if(train[index[0]][0][3] == testimage[3]):
		numcorrect = numcorrect + 1

print(numcorrect/len(test))



