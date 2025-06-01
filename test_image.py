import argparse
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import imageio
from models import resnet

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--iter', nargs='?', default=19000)
parser.add_argument('--gpu', nargs='?', default='false')
path_to_image = parser.parse_args().path
it = int(parser.parse_args().iter)
gpu = parser.parse_args().gpu
height, width = 3136, 4224
IMAGE_SIZE = height * width * 3
tf.compat.v1.disable_v2_behavior()
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if gpu == 'false' else None

x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, height, width, 3])

enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:
	iteration = it
	input(it)
	saver = tf.compat.v1.train.Saver()
	print('load saver')
	saver.restore(sess, f'models/kvadra_iteration_{it}.ckpt')
	#image = np.float16(cv2.imread(path_to_image))
	image = np.float16(np.array(Image.fromarray(imageio.imread(path_to_image)))) / 255
	image_2d = np.reshape(image, [1, IMAGE_SIZE])
	print('run test')
	enhanced_2d = sess.run(enhanced, feed_dict={x_: image_2d})
	enhanced_image = np.reshape(enhanced_2d, [height, width, 3]).clip(0, 1)
	enhanced_image = (enhanced_image * 255).astype(np.uint8)
	name = Path(path_to_name).stem
	print('save image')
	cv2.imwrite('visual_results/', f'enhanced_{name}.png')
	
