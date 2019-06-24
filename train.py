from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from time import time
import os

from resnet import resnet_layer


# 학습, 테스트 데이터셋 경로
TRAIN_DIR = os.path.join('./dataset/images/train')
TEST_DIR = os.path.join('./dataset/images/test')

# epoch : 학습 반복 횟수, steps_per_epoch : dataset의 수 / batch_size
epoch = 100
steps_per_epoch = 90
batch_size = 32

# 학습 데이터 Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# 학습, 테스트 데이터셋의 폴더 별로 자동 라벨링
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size=batch_size, target_size=(224, 224),
                                                    class_mode='categorical', color_mode='rgb')
test_generator = test_datagen.flow_from_directory(TEST_DIR, batch_size=batch_size, target_size=(224, 224),
                                                  class_mode='categorical', color_mode='rgb')

# number of classes
number_of_class = train_generator.num_classes


# 모델 파일이 있을 경우 불러옴
MODEL_SAVE_FOLDER_PATH = './model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

if os.listdir(MODEL_SAVE_FOLDER_PATH):
    file_list = os.listdir(MODEL_SAVE_FOLDER_PATH)
    file_name = file_list[0]
    print('load model : ' + file_name)
    model = load_model(MODEL_SAVE_FOLDER_PATH + file_name)
else:
    model = resnet_layer(number_of_class)

model_file_path = MODEL_SAVE_FOLDER_PATH + 'val_loss_{val_loss}-{val_acc:.4f}.hdf5'

# 학습 과정 설정
# Optimizer : Adam, Loss function : Categorical cross-entropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ModelCheckpoint
# verbose : 해당 함수의 진행 사항의 출력 여부, save_best_only : 모델의 정확도가 최고값을 갱신했을 때만 저장
checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=0, mode=min)

# TensorBoard
tensorBoard = TensorBoard(log_dir="logs/{}".format(time()))

callback_list = [checkpoint, tensorBoard]

# ImageDataGenerator로 얻어진 학습 데이터셋을 통한 학습은 fit() 대신 fit_generator()를 사용
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epoch, callbacks=callback_list,
                              validation_data=test_generator, validation_steps=number_of_class)


# 모델 저장
model.save('Clone_Classfication_model.hdf5')
print("Saving....")