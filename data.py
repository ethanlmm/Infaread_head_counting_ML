import os
import re
import time
import tensorflow as tf
import cv2
import numpy as np



#path
root=''
defult_model_path=root+'models/'
default_data_path=root+'data/'
image_path=default_data_path + 'image'

#types of files
jpg = '.*\.jpg|.*\.jpeg'
csv = '.*\.csv'
png = '.*\.png'




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
    if type is None: return list(map(lambda x: path + x, os.listdir(path)))
    paths=[]
    for file in os.listdir(path):
        if(re.match(type, file)):
            paths.append(path+file)
    return paths

def process(operation,obj_list): return list(map(operation,obj_list))




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







