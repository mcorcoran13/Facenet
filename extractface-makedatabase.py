
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot
#from mtcnn.mtcnn import MTCNN
from glob import glob
import os.path
#import rawpy, imageio
import cv2
import math

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
# 	""" 
# 	# load image from file
# 	image = Image.open(filename)
# 	# convert to RGB, if needed
# 	image = image.convert('RGB')
# 	# convert to array
# 	pixels = asarray(image)

#  """	
	# Read the input image
	img_orig = cv2.imread(filename)
	if 480 == img_orig.shape[1] and 480 == img_orig.shape[0]:
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
	
# """ 
# 	# create the detector, using default weights
# 	detector = MTCNN()
# 	# detect faces in the image
# 	results = detector.detect_faces(pixels)
# 	print(filename)
# 	print(results)

# 	if len(results) == 0:
# 		return
# 	print(results[0]['box'])
# 	print("---------------------")
# 	# extract the bounding box from the first face
# 	x1, y1, width, height = results[0]['box']
# 	# bug fix
# 	x1, y1 = abs(x1), abs(y1)
# 	x2, y2 = x1 + width, y1 + height
# 	# extract the face
# 	face = pixels[y1:y2, x1:x2]
# 	# resize pixels to the model size
# 	image = Image.fromarray(face)
# 	image = image.resize(required_size)
# 	face_array = asarray(image)
# 	return face_array
#  """
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
        #f.write('Possible Error: Double check file       '+i+'\n')
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
        #f.write('Error: no face found in file       '+i+'\n')
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
# 	""" plt.figure()
# 		plt.imshow(faces)
# 		plt.colorbar()
# 		plt.grid(False)	
# 		plt.show() """
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('./FaceDataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('./FaceDataset/val/')
# save arrays to one file in compressed format
savez_compressed('face-dataset.npz', trainX, trainy, testX, testy)
