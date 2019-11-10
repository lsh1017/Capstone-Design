import tensorflow as tf

MODEL_NAME = './model/Tea leaves/TeaLeaves_Xception'
MODEL_PATH = MODEL_NAME + '.hdf5'

converter = tf.lite.TFLiteConverter.from_keras_model_file(MODEL_PATH)

# tflite_model = tf.lite.toco_convert(input_data=MODEL_PATH, input_tensors=((299, 299, 3), 'float32'), output_tensors= )

# converter = tf.lite.TFLiteConverter.from_keras_model_file('xception.hdf5',                                                  input_shapes={'input_1':[1, 299, 299, 3]
tflite_model = converter.convert()

with open(MODEL_NAME + '.tflite', 'wb') as f:
    f.write(tflite_model)

