from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

from time import time
import os

from Xception import Xception
# from MobileNet import MobileNetv2
# from ResNet import resnet_layer
# from InceptionResnetV2 import inceptionResNetV2

Theme = 'Animals'

# 학습, 테스트 데이터셋 경로
TRAIN_DIR = os.path.join('./dataset/' + Theme + '/train')
TEST_DIR = os.path.join('./dataset/' + Theme + '/test')

MODEL_SAVE_DIR = './model/' + Theme + '/'
MODEL_SAVE_NAME = Theme + '_Xception.hdf5'
TENSORBOARD_NAME = Theme + ' ' + MODEL_SAVE_NAME

EPOCH = 10
BATCH_SIZE = 16
IMAGE_SIZE = (299, 299)

# 학습 데이터 Augmentation

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, rotation_range=90,
                                   zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='nearest', validation_split=0.2)
'''
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
'''

# 학습, 테스트 데이터셋의 폴더 별로 자동 라벨링
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE,
                                                    class_mode='categorical', color_mode='rgb', subset='training')
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE,
                                                  class_mode='categorical', color_mode='rgb', subset='validation')

number_of_class = train_generator.num_classes


model_file_path = MODEL_SAVE_DIR + 'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ' + MODEL_SAVE_NAME

if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)

# 학습 이어하기
# MODEL_FILE_DIR = ''
# if os.path.exists(MODEL_FILE_DIR):
#     print('load model : ' + MODEL_FILE_DIR)
#     model = load_model(MODEL_FILE_DIR)
# else:
#     model = Xception(number_of_class)

model = Xception(number_of_class)
# model.summary()

# 학습 과정 설정 / Optimizer : Adam, Loss function : Categorical cross-entropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ModelCheckpoint / verbose : 해당 함수의 진행 사항의 출력 여부, save_best_only : 모델의 정확도가 최고값을 갱신했을 때만 저장
checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=0, mode=min)

# TensorBoard
# Anaconda Prompt에 'tensorboard --logdir=logs' 입력
tensorBoard = TensorBoard(log_dir='./logs/{}'.format(TENSORBOARD_NAME + str(time())))

# Callbacks
callback_list = [checkpoint, tensorBoard]

model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, epochs=EPOCH, callbacks=callback_list,
                    validation_data=validation_generator, validation_steps=validation_generator.samples // BATCH_SIZE)

# 모델 저장
model.save(MODEL_SAVE_DIR + MODEL_SAVE_NAME)
print("Saving....")
