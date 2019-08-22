import re
import os


def path_generator(path, type=None):
    path=path + '/'
    if type is None: return list(map(lambda x: path + x, os.listdir(path)))
    paths=[]
    for file in os.listdir(path):
        if(re.match(type, file)):
            paths.append(path+file)
    return paths

def process(operation,obj_list): return list(map(operation,obj_list))

