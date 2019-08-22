from data import *
from model import *
from util import *
#one solution
images=FOR2(tf_jpg_read,path_generator(image_path,jpg))
print(images)
#other solution
images=FOR2(tf.io.read_file,path_generator(image_path,jpg))
images=FOR2(tf.io.decode_jpeg,images)

print(images)

