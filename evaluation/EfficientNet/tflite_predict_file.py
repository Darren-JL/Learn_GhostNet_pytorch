from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import resource_loader
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


import io
import os
import sys
import numpy as np
import cv2 as cv
import tensorflow as tf
import shutil
import json
import urllib3

http = urllib3.PoolManager(num_pools=10, maxsize=1000, timeout=urllib3.util.Timeout(connect=2.0, read=7.0))
guid2url = lambda guid, type_: ''.join(["http://pivot-dispatcher.momo.com/", type_, '/', guid[0:2], '/', guid[2:4], '/', guid, '_L.jpg'])


class InterpreterTest(test_util.TensorFlowTestCase):
    def __init__(self, tflite):
        super(InterpreterTest,self).__init__()
        self.tflitePath = tflite
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def init_interpreter(self):
        model_path = resource_loader.get_path_to_datafile(self.tflitePath)
        with io.open(model_path, 'rb') as model_file:
            data = model_file.read()

        self.interpreter = interpreter_wrapper.Interpreter(model_content=data)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        print(self.input_details)

        input_details = self.input_details
        self.assertEqual(1, len(input_details))
       // self.assertEqual('input', input_details[0]['name'])
        self.assertEqual(np.uint8, input_details[0]['dtype'])
       //self.assertTrue(([1, 224, 224, 3] == input_details[0]['shape']).all())//
       // self.assertEqual((0.007843137718737125, 128), input_details[0]['quantization'])

        self.output_details = self.interpreter.get_output_details()
        print(self.output_details)

        output_details = self.output_details
        self.assertEqual(1, len(output_details))
       //self.assertEqual('MobilenetV2/Predictions/Softmax', output_details[0]['name'])
        self.assertEqual(np.uint8, output_details[0]['dtype'])
        self.assertTrue(([1, 4] == output_details[0]['shape']).all())
        self.assertEqual((0.00390625, 0), output_details[0]['quantization'])

    def predict_image(self, image):
        test_input = image
        # self.interpreter.resize_tensor_input(self.input_details[0]['index'], test_input.shape)
        self.interpreter.resize_tensor_input(self.input_details[0]['index'], self.input_details[0]['shape'])
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], test_input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data[0]


if __name__ == '__main__':
    # interp = InterpreterTest('/mnt/data2/gyy/data/spam/1026/version2/mobilenet/quantize/0.9496_0.95/mobilenetv2_quantize/spam_mobilenet_v2_prun_quantize.tflite')
    tflite_path = sys.argv[1]
    interp = InterpreterTest(tflite_path)
    interp.init_interpreter()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "1"

    with tf.Graph().as_default() as g:
        file_input = tf.placeholder(tf.string, ())
        image = tf.image.decode_jpeg(tf.read_file(file_input), channels=3)
        images = tf.expand_dims(image, 0)
        # images = tf.cast(images, tf.float32) / 127.5 - 1
        images.set_shape((None, None, None, 3))
        images = tf.cast(tf.image.resize_images(images, (224, 224)), tf.uint8)

    sess = tf.Session(config=config, graph=g)

    # save_dir = '/home/momo/data2/gyy/data/spam/1026/tflite/code/avatar/'
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # fp = open('avatar-eval-densenet201_v2-online.txt')
    fp = open(sys.argv[3])
    lines = [f.strip('\n') for f in fp.readlines()]
    fp.close()

    label_map = {0: 'norm', 1: 'ad', 2: 'fuwu', 3: 'porn'}

    for m, line in enumerate(lines):
        info = json.loads(line)
        url = info['image']
        name = info['ground_truth']

        guid = url[url.rfind('/')+1:url.rfind('_')]
        url = guid2url(guid, 'album')

        raw = http.urlopen("GET", url).data
        img = cv.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv.IMREAD_COLOR)
        img = cv.resize(img, (224, 224))

        tmp_dir = save_dir + '/' + 'truth/' + name
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        img_path = tmp_dir + '/' + guid + '_L.jpg'
        cv.imwrite(img_path, img)

        print('img_path', img_path)

        try:
            input_uint8 = sess.run(images, feed_dict={file_input: img_path})
            imgs = np.expand_dims(img, 0)
            //out = interp.predict_image(imgs)# 输出的是int类型的，需要
        except Exception as e:
            print(e)
            continue

        label = out.argmax()
        pred_name = label_map[out.argmax()]
        prob = out.max() / 255.0

        if pred_name == 'fuwu' and prob < 0.8:
            pred_name  = 'fuwu_low'

        tmp_dir = save_dir + '/' + pred_name
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        shutil.copy(img_path, tmp_dir)

        print(m, label, prob, pred_name)

