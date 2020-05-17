import sys
sys.path.append('/home/momo/sun.zheng/GhostNet_tensorflow/EfficientNet/tpu/models/official/mnasnet')
sys.path.append('/home/momo/sun.zheng/GhostNet_tensorflow/EfficientNet/tpu/models/common')

from IPython import display
from PIL import Image
import numpy as np

filename = '/home/momo/sun.zheng/GhostNet_tensorflow/source/panda.jpg'
img = np.array(Image.open(filename).resize((224, 224))).astype(np.float)

import os
import tensorflow as tf

checkpoint_name = '/home/momo/sun.zheng/GhostNet_tensorflow/source/mnasnet-a1'
export_dir = os.path.join(checkpoint_name, 'saved_model')
serv_sess = tf.Session(graph=tf.Graph())
meta_graph_def = tf.saved_model.loader.load(serv_sess, [tf.saved_model.tag_constants.SERVING], export_dir)

import imagenet
import time

time_c = 0
for i in range(5000):
    print(i)
    time_start = time.time()
    top_class, probs = serv_sess.run(fetches=["ArgMax:0", "softmax_tensor:0"], feed_dict={"Placeholder:0": [img]})
    time_end = time.time()
    time_tmp = time_end - time_start
    time_c = time_c + time_tmp

print('time cost:' + str(time_c) + 's')

#print("Top class: ", top_class[0], " with Probability= ", probs[0][top_class[0]])
#label_map = imagenet.create_readable_names_for_imagenet_labels()  
#for idx, label_id in enumerate(reversed(list(np.argsort(probs)[0][-5:]))):
    #print("Top %d Prediction: %d, %s, probs=%f" % (idx+1, label_id, label_map[label_id], probs[0][label_id]))





