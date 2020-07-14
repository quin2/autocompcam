"""
requires compCam, pcaMode, and ssModel files!
the ones in the repo were made on Google Cloud, so if this doesn't work try
remaking them on jupyter on your machine.
"""

import numpy as np 
import cv2

from joblib import load
import math

#helper: split image into grid
def gridSplit(im, M, N):
    return [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

#helper: make SPSM
def histogram2(ssm):
    #find height and width
    height = ssm.shape[0]
    width = ssm.shape[1]
    
    #level 0
    hist = list(np.bincount(ssm.ravel(),minlength=256))
    
    #level 1
    sGrid = gridSplit(ssm, math.ceil(height/2), math.ceil(width/2))
    histograms = [list(np.bincount(sub.ravel(),minlength=256)) for sub in sGrid]
    final = []
    for idx, itm in enumerate(histograms[0]):
        for arr in histograms:
            final.append(arr[idx])
    hist.extend(final)
  
    #level 2
    sGrid = gridSplit(ssm, math.ceil(height/4), math.ceil(width/4))
    histograms = [list(np.bincount(sub.ravel(),minlength=256)) for sub in sGrid]
    final = []
    for idx, itm in enumerate(histograms[0]):
        for arr in histograms:
            final.append(arr[idx])
    hist.extend(final)

    return np.array(hist)

#define pipeline
def pipeline(img):    
    #saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create() 
    #saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    ssm = (saliencyMap * 255).astype("uint8")

    
    #histogram
    hist = histogram2(ssm)
    rhist = hist.reshape(1, -1)
    
    #ss and pca
    rhist = ss.transform(rhist)
    rhist = pca.transform(rhist)

    return rhist

#load model
regr = load('models/compCam.joblib') 
ss = load('models/ssModel.joblib')
pca = load('models/pcaMode.joblib')

#start video capture
cap = cv2.VideoCapture(0)

#count frames so we don't bog down CV
ctr = 0
maxf = 5

guess = 0.0

while(True):
	ret, frame = cap.read()

	#set dimensions to 100x100
	ret = cap.set(3, 400)
	ret = cap.set(4, 400)

	#find composition score!
	#if ctr > maxf:
	pframe = pipeline(frame)
	guess = regr.predict(pframe)

	#	ctr = 0

	font = cv2.FONT_HERSHEY_SIMPLEX 
	new = cv2.putText(frame, str(guess), (50,50), font,  
                   1, (255, 0, 0), 2, cv2.LINE_AA) 

	cv2.imshow("autoCompCam", new)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

	#ctr += 1

cap.release()
cv2.destroyAllWindows()