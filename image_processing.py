import cv2
import os


IMAGE_FOLDER_DIR = '.\\dataset_for_image_processing\\original_images\\'
RESIZE_RATIO = 0.2


def image_prossing(dir):
    try:
        for path, dir, files in os.walk(dir):
            if not os.path.exists(path.replace('original', 'processed')):
                os.mkdir(path.replace('original', 'processed'))

            for file in files:
                ext = os.path.splitext(file)[-1]
                if ext == '.jpg' or ext == '.JPG':
                    image_path = os.path.join(path, file)
                    processed_image_path = os.path.join(path.replace('original', 'processed'), file)
                    calculateSize(image_path, processed_image_path)
    except PermissionError:
        pass


def calculateSize(file, image_save_path):
    image = cv2.imread(file)
    image = cv2.resize(image, dsize=None, fx=RESIZE_RATIO, fy=RESIZE_RATIO)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image_temp = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)


    width = image_temp.shape[1]
    height = image_temp.shape[0]

    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0

    # y_max
    detect = False
    for y in range(height):
        if detect == True:
            break
        for x in range(width):
            if image_temp[y, x] == 0:
                y_max = y
                detect = True
                break

    # y_min
    detect = False
    for y in range(height - 1, 0, -1):
        if detect == True:
            break
        for x in range(width):
            if image_temp[y, x] == 0:
                y_min = y
                detect = True
                break

    # x_min
    detect = False
    for x in range(width):
        if detect == True:
            break
        for y in range(height):
            if image_temp[y, x] == 0:
                x_min = x
                detect = True
                break

    # x_max
    detect = False
    for x in range(width - 1, 0, -1):
        if detect == True:
            break
        for y in range(height):
            if image_temp[y, x] == 0:
                x_max = x
                detect = True
                break

    if x_min >= 10:
        x_min -= 10
    if (x_max + 10) <= width:
        x_max += 10
    if y_max >= 10:
        y_max -= 10
    if (y_min + 10) <= height:
        y_min += 10

    image = image[y_max:y_min, x_min:x_max]

    # cv2.imshow('image', image)
    # cv2.waitKey(1)

    cv2.imwrite(image_save_path, image)
    print("Save :", image_save_path)

image_prossing(IMAGE_FOLDER_DIR)