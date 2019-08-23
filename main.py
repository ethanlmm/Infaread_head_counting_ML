from data import *
from model import *
from util import *
import tensorflow as tf
from tensorflow import keras

mcnn = MCNN()
dilation=CrowdNet()

input=keras.layers.Input(shape=(None,None,3))
out=mcnn(input)
out=dilation(out)

model=keras.Model(input,out,name='IR')
model.summary()


