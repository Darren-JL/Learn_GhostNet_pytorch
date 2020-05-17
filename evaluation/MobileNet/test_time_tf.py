import numpy as np
import PIL
import tensorflow as tf
from datasets import imagenet
import time
import os


#file_name = os.listdir('/mnt/data2/raw/val_imagenet') # 验证集50000张图像

time_c = 0
checkpoint_name = '/home/sz/MobileNet/mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant'
#count = 0
#N = 5000 # 只算5000张图像

img = np.array(PIL.Image.open('/home/sz/MobileNet/panda.jpg').resize((224, 224))).astype(np.float) / 128 - 1
gd = tf.GraphDef.FromString(open(checkpoint_name + '_frozen.pb', 'rb').read())
inp, predictions = tf.import_graph_def(gd,  return_elements = ['input:0', 'MobilenetV1/Predictions/Reshape_1:0'])

with tf.Session(graph=inp.graph):
    for i in range(5000):
        print(i)
        time_start = time.time()
        x = predictions.eval(feed_dict={inp: img.reshape(1, 224,224, 3)})
        time_end = time.time()
        time_tmp = time_end - time_start
        time_c = time_c + time_tmp

#label_map = imagenet.create_readable_names_for_imagenet_labels()  
#print("Top 1 Prediction: ", x.argmax(),label_map[x.argmax()], x.max())
print('time cost:', time_c, 's')

