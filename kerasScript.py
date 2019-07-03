import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import json
import cv2
import gc

def classification(imagePath):
	model = load_model('model.hdf5')

	image = cv2.imread(imagePath)
	image = cv2.resize(image, (224, 224))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	output = model.predict(image)[0]
	np.set_printoptions(formatter={'float': lambda y: "{0:0.3f}".format(y)})

	print(output)
	del model
	gc.collect()
	return {
		'success': 1,
		'result': str(output)
	}


