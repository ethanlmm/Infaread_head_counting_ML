import os
import re
import time
import tensorflow as tf
import cv2
import numpy as np

# path
root = ''
defult_model_path = root + 'models'
default_data_path = root + 'data'
shanghai_path = default_data_path + '/ShanghaiTech_Crowd_counting_Dataset'

main_train_image = shanghai_path + '/part_B_final/train_data/images'
main_test_den = shanghai_path + '/part_B_final/train_data/ground_truth'

# types of files
jpg = '.*\.jpg|.*\.jpeg'
csv = '.*\.csv'
png = '.*\.png'
mat = '.*\.mat'

dataset = "B"

train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
train_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den/'
train_den_quater_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den_quarter/'
val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'
val_den_quater_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den_quarter/'
img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'


def load_model(path):
    with  open(path + '/model.json', 'r') as file:
        loaded_model_json = file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(path + '/weight.h5')
    return loaded_model


def save_model(model, path=defult_model_path):
    tm = path + time.strftime("%Y-%m-%d_%H_%M", time.gmtime(time.time()))
    os.mkdir(tm)
    model.save_weight(path + "weight.h5")
    model_js = model.to_json()
    with open(path + '/' + "model.json", "w") as file:
        file.write(model_js)


def tf_img_read(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, dtype=tf.float32)
    img = img / 255.0

    # add preprocess step here
    return img


def path_generator(path, type=None):
    if type is None: return [os.path.join(path, name) for name in os.listdir(path)]
    paths = []
    for name in os.listdir(path):
        if (re.match(type, name)):
            paths.append(os.path.join(path, name))
    return paths


def FOR(operation, obj_list): return list(map(operation, obj_list))


def FOR2(operation, obj_list): return [operation(x) for x in obj_list]


def csv_read(path):
    csv = np.loadtxt(path, delimiter=',')
    csv = np.reshape(csv, (csv.shape[0], csv.shape[1], 1))
    return csv


def cv2_image_read(path):
    img = cv2.imread(path, 0)
    img = np.array(img)
    img = (img - 127.5) / 128
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img
