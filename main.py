from data import *
from model import *
from util import *
import tensorflow as tf


mcnn = MCNN()
dilation=CrowdNet()
mcnn.summary()
dilation.summary()

model=tf.keras.Sequential([dilation,
                            mcnn])



#one solution
images=process(tf_jpg_read,path_generator(image_path,jpg))
print(images)
#other solution
images=process(tf.io.read_file,path_generator(image_path,jpg))
images=process(tf.io.decode_jpeg,images)

print(images)

