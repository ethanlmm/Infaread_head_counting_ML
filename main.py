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



