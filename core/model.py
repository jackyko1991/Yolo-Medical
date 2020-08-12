import tensorflow as tf
import datetime
from core import utils
from core import transforms
from core import NiftiDataset2D
from core import NiftiDataset3D
import numpy as np

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
		self.training_label_filename = self.config['TrainingSetting']['Data']['TrainingLabelFilename']
		self.testing_label_filename = self.config['TrainingSetting']['Data']['TestingLabelFilename']
		self.class_names_filename = self.config['TrainingSetting']['Data']['ClassNamesFilename']
		self.datatype = self.config['TrainingSetting']['Data']['Type']

		self.trainingbatch_size = self.config['TrainingSetting']['BatchSize']
		self.image_log = self.config['TrainingSetting']['ImageLog']
		self.testing = self.config['TrainingSetting']['Testing']
		self.test_step = self.config['TrainingSetting']['TestStep']

		self.restore_training = self.config['TrainingSetting']['Restore']
		self.log_dir = self.config['TrainingSetting']['LogDir']
		self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']
		self.epochs = self.config['TrainingSetting']['Epochs']
		self.max_itr = self.config['TrainingSetting']['MaxIterations']
		self.log_interval = self.config['TrainingSetting']['LogInterval']

		# self.network_name = self.config['TrainingSetting']['Networks']['Name']
		# self.dropout_rate = self.config['TrainingSetting']['Networks']['Dropout']
		self.grid_size = self.config['Networks']['GridSize']
		self.bounding_boxes_per_cell = self.config['Networks']['BoundingBoxesPerCell']
		self.patch_shape = self.config['Networks']['PatchShape']
		self.dimension = len(self.patch_shape)

		# self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
		# self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
		# self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
		# self.decay_steps = self.config['TrainingSetting']['Optimizer']['Decay']['Steps']
		# self.spacing = self.config['TrainingSetting']['Spacing']
		# self.drop_ratio = self.config['TrainingSetting']['DropRatio']
		# self.min_pixel = self.config['TrainingSetting']['MinPixel']

		# self.loss_name = self.config['TrainingSetting']['Loss']

		# # evaluation config
		# self.checkpoint_path = self.config['EvaluationSetting']['CheckpointPath']
		# self.evaluate_data_dir = self.config['EvaluationSetting']['Data']['EvaluateDataDirectory']
		# self.evaluate_image_filenames = self.config['EvaluationSetting']['Data']['ImageFilenames']
		# self.evaluate_label_filename = self.config['EvaluationSetting']['Data']['LabelFilename']
		# self.evaluate_probability_filename = self.config['EvaluationSetting']['Data']['ProbabilityFilename']
		# self.evaluate_stride = self.config['EvaluationSetting']['Stride']
		# self.evaluate_batch = self.config['EvaluationSetting']['BatchSize']
		# self.evaluate_probability_output = self.config['EvaluationSetting']['ProbabilityOutput']

		print("{}: Reading configuration file complete".format(datetime.datetime.now()))

	def dataset_iterator(self,label_filename,transforms,train=True):
		if self.dimension == 2:
			Dataset = NiftiDataset2D.NiftiDataset(
				label_filename = self.training_label_filename,
				transforms = transforms,
				train=True,
				class_num=self.class_num
				)

		dataset = Dataset.get_dataset()
		if self.dimension == 2:
			dataset = dataset.shuffle(buffer_size=5)

		dataset = dataset.batch(self.batch_size,drop_remainder=True)

		return dataset.make_initializable_iterator()


	def build_model_graph(self):
		print("{}: Start to build model graph...".format(datetime.datetime.now()))

		self.global_step_op = tf.train.get_or_create_global_step()

		self.classnames = utils.read_class_names(self.class_names_filename)

		if self.dimension == 2:
			input_batch_shape = (None,self.patch_shape[0],self.patch_shape[1], 1)
			output_batch_shape = (None,self.grid_size,self.grid_size,self.bounding_boxes_per_cell*5+len(self.classnames))
		elif self.dimension == 3:
			input_batch_shape = (None,self.patch_shape[0],self.patch_shape[1],self.patch_shape[2], 1)
			output_batch_shape = (None,self.grid_size,self.grid_size,self.grid_size,self.bounding_boxes_per_cell*5+len(self.classnames))
		else:
			sys.exit('Invalid Patch Shape (length should be 2 or 3)')

		self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(input_batch_shape,output_batch_shape)

		# Get images and labels
		# create transformations to images and labels
		# Force input pipeline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow
		with tf.device('/cpu:0'):
			if self.dimension == 2:
				train_transforms = transforms.transform_2d(self.patch_shape)
			# elif self.dimension == 3:
				# train_transforms = transforms.transform_3d

			# get input and output datasets
			self.train_iterator = self.dataset_iterator(self.training_label_filename,train_transforms)
			self.next_element_train = self.train_iterator.get_next()

	def train(self):
		# read config to class variables
		self.read_config()

		""" Setup network model """
		self.build_model_graph()

		# initialize all variables
		self.sess.run(tf.initializer.global_variables())
		print("{}: Start training...".format(datetime.datetime.now()))

		# training cycle, loop over epochs
		for epoch in np.arange(0, self.epochs):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))

			# initialize iterator in each new loop
			self.sess.run(self.train_iterator.initializer)

			while True:
				try:
					self.sess.run(tf.local_variables_initializer())

					image, label = self.sess.run(self.next_element_train)
				except tf.errors.OutOfRangeError:
					break
