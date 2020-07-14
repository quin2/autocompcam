import json
from flask import Flask, jsonify, Response, request, current_app

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
	(success, saliencyMap) = saliency.computeSaliency(img)
	ssm = (saliencyMap * 255).astype("uint8")


	#histogram
	hist = histogram2(ssm)
	rhist = hist.reshape(1, -1)

	#ss and pca
	rhist = current_app.ss.transform(rhist)
	rhist = current_app.pca.transform(rhist)

	return rhist

app = Flask(__name__)
app.regr = load('models/compCam.joblib')
app.ss = load('models/ssModel.joblib')
app.pca = load('models/pcaMode.joblib')

@app.route('/')
def index():
    return "usage: send picture to /v1/score to evaluate scene composition"

@app.route('/v1/score', methods=['POST'])
def compScore():
	#load from request
	data = request.files['file']
	npimg = np.fromfile(data, np.uint8)
	img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

	#predict
	pframe = pipeline(img)
	guess = current_app.regr.predict(pframe)

	return jsonify(
        score=guess[0]
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=80)
