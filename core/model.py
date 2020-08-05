import tensorflow as tf

class Model(object):
	def __init__(self, sess, config):
		"""
		Args:
			sess: Tensorflow session
			config: Model configuration
		"""

		self.sess = sess
		self.config = config
		self.model = None
		self.graph = tf.Graph()
		self.graph.as_default()
		self.epoches = 99999999999999999

	def read_config(self):
		print("{}: Reading configuration file...".format(datetime.datetime.now()))

		# training config
		

	def build_model_graph():
		print("{}: Start to build model graph...".format(datetime.datetime.now()))

		self.global_step_op = tf.train.get_or_create_global_step()



	def train(self):
		""" Setup network model """
		self.build_model_graph()