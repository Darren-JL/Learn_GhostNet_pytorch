import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第0块gpu为可见



# Load TFLite model and allocate tensors.

# EfficientNet uint8模型
interpreter = tf.lite.Interpreter(model_path='./efficientnet-lite2/efficientnet-lite2-int8.tflite')

# EfficientNet fp32模型
#interpreter = tf.lite.Interpreter(model_path='./efficientnet-lite2/efficientnet-lite2-fp32.tflite')

# MobileNet
#interpreter = tf.lite.Interpreter(model_path='/home/momo/sun.zheng/GhostNet_tensorflow/source/v3-large_224_1.0_uint8/v3-large_224_1.0_uint8.tflite')

# Mnasnet
#interpreter = tf.lite.Interpreter(model_path='/home/momo/sun.zheng/GhostNet_tensorflow/source/mnasnet-a1/mnasnet-a1.tflite')

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

'''
# 测试单张图像
#img = Image.open("./input.png")
img = Image.open('/mnt/data2/raw/val_imagenet/ILSVRC2012_val_00020421.JPEG').resize((240,240))
#print(img)
img = np.array(img, np.uint8)
#print(img.shape)
img = np.expand_dims(img, axis=0)
#print(img.shape)


interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

label_pre = output_data.argmax()
print('label_pre:' + str(label_pre))
#prob = output_data.max() / 255.0 # 概率
#print(prob)
'''


# 测试5000张图像
file_path = '/mnt/data2/raw/val_imagenet/'
label_path = '/mnt/data2/raw/caffe_ilsvrc12/val.txt'

'''
# 将val.txt中的标签抽取出来，单独形成一个数组
label = np.zeros((50000,), dtype=np.int) # 构造标签数组
n = 0

with open(label_path, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        #print(type(line)) # str
        length = len(line)
        index = -(length - 29) # 决定截取后面的标签
        label_current = line[index:]
        #print(type(label_current)) # str
        label[n] = int(label_current)
        n = n + 1
np.savetxt('/mnt/data2/raw/caffe_ilsvrc12/val_label.txt', label, fmt="%d") # 将标签保存为十进制的，不是科学计数法
'''

# 构造标签字典，文件名为键，标签为值
with open(label_path, 'r') as f:
    label_dict = []
    for line in f.readlines():
        line = line.strip('\n') # 去掉换行符\n
        b = line.split(' ') # 将每一行以空格为分隔符转换为列表
        label_dict.append(b)

label_dict = dict(label_dict)
#print(type(label_dict['ILSVRC2012_val_00050000.JPEG']))



time_c = 0 # 累计时间
N = 0 # 实际测试的图像张数
count = 0 # 当前测试图片的张数
N_right = 0 # 测试正确的图片张数

for file in os.listdir(file_path):
    count = count + 1
    print(str(count) + ':'  + file)
    img = Image.open(file_path + file).resize((224, 224))
    img = np.array(img, np.uint8)
    #img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, axis = 0)
    if img.ndim!=4:
        continue
    N = N + 1 # 实际测试的图像张数
    # 测试时间
    time_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    time_end = time.time()
    time_tmp = time_end - time_start
    time_c = time_c + time_tmp
    
    label_pre = output_data.argmax()
    print('label_pre:' + str(label_pre))
    print('label:' + str(label_dict[file]))
    if int(label_pre) == int(label_dict[file]):
        N_right = N_right + 1
        print('right')

    print('count:' + str(count))
    print('N_right:' + str(N_right))
    print('N:' + str(N))
    if count == 500:
        break

print('time cost:' + str(time_c) + 's')
print('val acc:' + str(N_right / N))






