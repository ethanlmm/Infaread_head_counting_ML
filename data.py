import os
import re
import time
import tensorflow as tf
import cv2
import numpy as np
import scipy.io as sio



#path
root=''
defult_model_path=root+'models'
default_data_path=root+'data'
shanghai_path=default_data_path+'/ShanghaiTech_Crowd_counting_Dataset'

main_train_image=shanghai_path+'/part_B_final/train_data/images'
main_test_den=shanghai_path+'/part_B_final/train_data/ground_truth'

#types of files
jpg = '.*\.jpg|.*\.jpeg'
csv = '.*\.csv'
png = '.*\.png'
mat = '.*\.mat'




def load_model(path):
    with  open(path+'/model.json', 'r') as file:
        loaded_model_json = file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(path+'/weight.h5')
    return loaded_model

def save_model(model,path=defult_model_path):
    tm=path+time.strftime("%Y-%m-%d_%H_%M", time.gmtime(time.time()))
    os.mkdir(tm)
    model.save_weight(path+"weight.h5")
    model_js=model.to_json()
    with open(path+'/'+"model.json","w") as file:
        file.write(model_js)
def tf_jpg_read(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw)
    # add preprocess step here
    return img

def tf_png_read(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw)
    # add preprocess step here
    return img

def path_generator(path, type=None):
    path=path + '/'
    if type is None: return [path+name for name in os.listdir(path)]
    paths=[]
    for file in os.listdir(path):
        if(re.match(type, file)):
            paths.append(path+file)
    return paths

def FOR(operation,obj_list): return list(map(operation,obj_list))
def FOR2(operation,obj_list):return [operation(x) for x in obj_list]

def mat_read(path):
    mat_contents = sio.loadmat(path)
    return mat_contents

#belows are given, may not be optimized
def cv2_image_read(path):
    img=cv2.imread(path)
    img=np.array(img)
    img=img/255.0
    return img

def csv_read(path):
    with open(path) as file:
        den = np.loadtxt(file, delimiter = ",")
    return den

def den_quater(matrix):
    den_quarter = np.zeros((int(matrix.shape[0] / 4), int(matrix.shape[1] / 4)))
    ...







