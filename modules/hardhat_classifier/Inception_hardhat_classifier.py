import time
import numpy as np
import tensorflow as tf

class InseptionHardhatClassifier:
	def __init__(self):

		PATH_TO_PB = './models/hardhat_classifier/inception_hardhats_model.pb'

		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		with self.detection_graph.as_default():
			config = tf.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = 0.15
			self.sess = tf.Session(graph=self.detection_graph, config=config)

		self.image_tensor = self.detection_graph.get_tensor_by_name('input_1:0')
		self.classes = self.detection_graph.get_tensor_by_name('dense_1/Softmax:0')

		self.classes_dict = {1: 'nothing', 0: 'hardhat'}


	def run(self, image):
		#image_np = cv2.resize(box_image, (299,299))
		image_np_expanded = np.expand_dims(image, axis=0)

		(classes) = self.sess.run(
				[self.classes],
				feed_dict={self.image_tensor: image_np_expanded})
		classes = self.classes_dict[np.argmax(classes)]
		
		return classes