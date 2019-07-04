from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing.image import img_to_array
import os
from PIL import Image
import cv2

MODEL_FILE_PATH = './Xception_model.h5'
# MODEL_FILE_PATH = './val_loss=1.3712, val_acc=0.6526.hdf5'
# TEST_DIR = os.path.join('./dataset/images/test')
IMAGE_PATH = './dataset/processed_images/test/GMB11/_MG_9628.JPG'

model = load_model(MODEL_FILE_PATH)

image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (299, 299))
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

output = model.predict(image)[0]
np.set_printoptions(formatter={'float': lambda y: '{0:0.3f}'.format(y)})

print(output)

# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(299, 299), batch_size=16, class_mode='categorical')
#
# print("-- Evaluate --")
# scores = model.evaluate_generator(test_generator, steps=50)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
#
# print("-- Predict --")
# # output = model.predict_generator(test_generator, steps=5)
# output = model.predict_generator(test_generator, steps=5)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#
# print(test_generator.class_indices)
# print(output)