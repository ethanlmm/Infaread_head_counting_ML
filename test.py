from data import *
from model import *
from util import *

images=FOR(tf_jpg_read,path_generator(main_train_image,jpg))
print(len(images))
dens =FOR(mat_read,path_generator(main_test_den,mat))


