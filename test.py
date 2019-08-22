from data import *
from model import *
from util import *
#one solution
images=process(tf_jpg_read,path_generator(image_path,jpg))
print(images)
#other solution
images=process(tf.io.read_file,path_generator(image_path,jpg))
images=process(tf.io.decode_jpeg,images)

print(images)

