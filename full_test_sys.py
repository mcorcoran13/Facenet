from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import pickle
from os import listdir
from os.path import isdir
from glob import glob
import os.path
import cv2
import math

#Embedding part
#///////////////////////////////////////////////////////////////////////////////
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


#height and width of output
width_, height_  = 160, 160
# this allows me to include more area around the face (multipication factor)
exp_x,exp_y = 1.3,1.3

#load face finding classifiers
#///////////////////////////////////////////////////////////////////////////////
Classifier = [
'haarcascade_frontalface_default.xml',  #0
'haarcascade_frontalface_alt.xml',      #1
'haarcascade_frontalface_alt2.xml',     #2
'haarcascade_frontalface_alt_tree.xml', #3
'haarcascade_eye.xml',                  #4
'haarcascade_eye_tree_eyeglasses.xml',  #5
'haarcascade_lefteye_2splits.xml',      #6
'haarcascade_righteye_2splits.xml',     #7
'haarcascade_profileface.xml']          #8
#///////////////////////////////////////////////////////////////////////////////

# Load the cascade
face_cascade = cv2.CascadeClassifier('./XML_classifiers/'+Classifier[2])
profile_cascade = cv2.CascadeClassifier('./XML_classifiers/'+Classifier[8])

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):

	# Read the input image
	img_orig = cv2.imread(filename)
	if img_orig.shape[1] == img_orig.shape[0]:
		img_resized = cv2.resize(img_orig, (width_,height_), interpolation = cv2.INTER_AREA)
		face_array = asarray(img_resized)
		return face_array
	else:
		flipped = False
		faces,img,width,height = findOneface(img_orig)

		# this crops the face out and scales it as a box
		for (x,y,w,h) in faces:
			#locate center of box
			x_cent, y_cent = (x+(w//2)), (y+(h//2))
			#establish x,y,w,h for scale factors
			w2, h2 = int(math.floor(w*exp_x)), int(math.floor(h*exp_y))
			x2, y2 = (x_cent-(w2//2)), (y_cent-(h2//2))
			#check if it's to close to the edge of the frame
			if (x2 < 0 or x2+w2 > width) or (y2 < 0 and y2+h2 > height):
				x2, y2 = x, y
				w2, h2 = w, h
			#produce cropped image of face
			crop_img = img[y2:y2+h2, x2:x2+w2]
			crop_resized = cv2.resize(crop_img, (width_,height_), interpolation = cv2.INTER_AREA)
			#cv2.imshow("cropped", crop_resized)
			print('successfully found face')
			if flipped == False:
				face_array = asarray(crop_resized)
				return face_array
			else:
				print("image was flipped back")
				face_array = asarray(cv2.flip(crop_resized))
				return face_array
	return None

def reduceSize(img, scale_factor=0.5):
    width = img.shape[1]
    height = img.shape[0]
    if (img.shape[0] >= 1500) or (img.shape[1] >= 1500):
        width = int(math.floor(img.shape[1]*scale_factor))
        height = int(math.floor(img.shape[0]*scale_factor))
        img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    elif scale_factor != 0.5:
        width = int(math.floor(img.shape[1]*scale_factor))
        height = int(math.floor(img.shape[0]*scale_factor))
        img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    return img,width,height

def findOneface(img_orig):
    global flipped
    img,width,height = reduceSize(img_orig)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) > 1:
        img,width,height = reduceSize(img_orig,0.3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        if len(faces) < 1:
            faces = profile_cascade.detectMultiScale(gray, 1.1, 6)
            if len(faces) < 1:
                img,width,height = reduceSize(img_orig)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) < 1:
        faces = profile_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) < 1:
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = profile_cascade.detectMultiScale(gray, 1.1, 6)
        flipped = True
        print("image was flipped once")
        if len(faces) < 1:
            faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) < 1:
        print('Error: no face found in file')
    return faces,img,width,height

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		if face is not None:
			faces.append(face)
	return asarray(faces)


# get the face embedding for one face
def get_embedding(model_embed, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model_embed.predict(samples)
	return yhat[0]

# load saved model
filename = 'face_classifier.sav'
model = pickle.load(open(filename, 'rb'))

# load the facenet model
model_embed = load_model('facenet_keras.h5')
print('Loaded Model')

# load train dataset
testX_faces = load_faces('./test_images/')

# convert each face in the test set to an embedding
testX = list()
for face_pixels in testX_faces:
	embedding = get_embedding(model_embed, face_pixels)
	testX.append(embedding)
testX = asarray(testX)
print(testX.shape)

data = load('face-embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
#testy = out_encoder.transform(testy)


# test model on input images
#selection = choice([i for i in range(testX.shape[0])])
print("The number of test images found was ", testX.shape[0])
selection = int(input('Enter a viable index to view the corresponding prediction (enter -1 to quit): '))

while selection >= 0:
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    #random_face_class = testy[selection]
    #random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    #print('Expected: %s' % random_face_name[0])
    # plot for fun
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()
    selection = int(input('Enter an index for the test image: '))

# restore np.load for future normal usage
np.load = np_load_old
