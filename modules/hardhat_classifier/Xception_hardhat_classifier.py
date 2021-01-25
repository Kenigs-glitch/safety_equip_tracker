import time
import numpy as np
import tensorflow as tf

class XceptionHardhatClassifier:
	def __init__(self):

		PATH_TO_PB = './modules/hardhat_classifier/Xception_hardhat_with_hats.pb'

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

		self.classes_dict = {0: 'nothing', 1: 'hardhat'}


	def run(self, image):
		#image_np = cv2.resize(box_image, (299,299))
		image_np_expanded = np.expand_dims(image, axis=0)

		(classes) = self.sess.run(
				[self.classes],
				feed_dict={self.image_tensor: image_np_expanded})

		print(f"Hardhats : {classes[0][0][1]}")
		clas = 'nothing'
		if classes[0][0][1] > 0.85:
			clas = 'hardhat'
			
		#classes = self.classes_dict[np.argmax(classes)]
		
		return clas