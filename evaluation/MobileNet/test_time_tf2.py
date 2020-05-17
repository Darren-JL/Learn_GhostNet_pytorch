import tensorflow as tf
from nets import mobilenet_v1
from nets.mobilenet import mobilenet_v2
from nets.mobilenet import mobilenet_v3

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))


# 测试MobileNetV2,MobileNetV3
# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobilenet_v3.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v3.mobilenet(images)

# 测试MobileNetV1
#with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
    #logits, endpoints = mobilenet_v1.mobilenet_v1(images)
        # Restore using exponential moving average since it produces (1.5-2%) higher 
        # accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)


from datasets import imagenet
import os
import time

# 加载模型
#checkpoint = '/home/momo/sun.zheng/GhostNet_tensorflow/source/mobilenet_v1_1.0_224.ckpt' # v1-float32
#checkpoint = '/home/momo/sun.zheng/GhostNet_tensorflow/source/mobilenet_v1_1.0_224_quant.ckpt' # v1-uint8

#checkpoint = '/home/momo/sun.zheng/GhostNet_tensorflow/source/mobilenet_v2_1.0_224.ckpt' # v2-float32

checkpoint = '/home/momo/sun.zheng/GhostNet_tensorflow/source/v3-large_224_1.0_float/pristine/model.ckpt-540000' # v3-float32
#checkpoint = '/home/momo/sun.zheng/GhostNet_tensorflow/source/v3-large_224_1.0_uint8/pristine/model.ckpt-2790692' # v3-uint8


time_c = 0
# 同一张图像测5000次，测量时间
pic = '/mnt/data2/raw/val_imagenet/ILSVRC2012_val_00041670.JPEG'

with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    time_start = time.time()
    for i in range(5000):
        print(i)
        x = endpoints['Predictions'].eval(feed_dict={file_input: pic})
    time_end = time.time()
    time_c = time_end - time_start

#x = endpoints['Predictions'].eval(feed_dict={file_input: 'panda.jpg'})
#label_map = imagenet.create_readable_names_for_imagenet_labels()  
#print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
print('time cost:', time_c, 's')


'''
# 测五千张图像，测量时间
pic_path = '/mnt/data2/raw/val_imagenet/' #  测试集五万张图像的路径
i = 0 #  当前图像张数
N = 5000 # 只测试5000张

for img in os.listdir(pic_path):
    i = i + 1
    print(i)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        time_start = time.time()
        x = endpoints['Predictions'].eval(feed_dict={file_input: pic_path + img})
        time_end = time.time()
        time_tmp = time_end - time_start
        time_c= time_c + time_tmp
    print(time_c)
    if i>=5000:
        break

print('time cost:', time_c, 's')
'''












