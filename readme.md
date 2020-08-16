# autoCompCam

My work implementing the support vector regressor found in [this paper](http://fangchen.org/paper_pdf/FLMS_mm14.pdf): *Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models*.  
More information can be found in my [blog post](https://quinnv.space/writing/autocompcam.html).

* Running real-time example
Install requirements.txt, and use python3 to run `webcam_demo.py`. You'll need something with an Intel processor because of the way Pickle works, and you'll want to have a webcam connected to your machine.

* Notebook (autocompcam.ipynb)
Install requirements.txt. The main author of the paper I linked has a website where you can download the `release_data` directory with the crops in matlab file format, as well as the training images they used. I set things up so the directory with the uncropped images is in the `release_data` directory. 

* API 
I deployed this on heroku, the main endpoint is pretty simple. POST the image you want to evaluate to `https://autocompcam.herokuapp.com/v1/score` with the image data set to the `file` key and `content-type` set to `multipart/form-data`. YMMV, I'll get an example working at some point. It might be a little slow. If you want to setup the service yourself, use the procfile here, but be aware you'll have to set up your dyno to take in `Aptfile` and install OpenCV that way. 