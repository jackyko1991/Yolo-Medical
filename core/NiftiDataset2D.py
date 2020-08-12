import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import random

def NiftiDataset(object):
	"""
	Load image-label pair for training, testing and inference.
	Args:
		label_filename (string): Path to label text file.
		transforms (list): List of SimpleITK image transformations.
		class_num (int): Number of output classes
		train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
	"""

	def __init__(self,
		label_filename = '',
		transforms=None,
		train=False,
		class_num=1
		):

		# initialize membership variables
		self.label_filename = label_filename
		self.transforms = transforms
		self.train = train
		self.class_num = class_num

		def get_dataset(self):
			with open(self.label_filename,'r') as f:
				txt = f.readlines()
				annotations = [line.strip() for line in txt if len(line.strip().split()[2:] != 0)]

			# randomize the annotations
			random.shuffle(annotations)

			annotation_list = [annotation.strip().split()[0] for annotation in annotations]
			slice_num_list = [annotation.strip().split()[1] for annotation in annotations]
			bboxes_list = [annotation.strip().split()[2:] for annotation in annotations]

			dataset = tf.data.Dataset.from_tensor_slices((filename_list,slice_num_list,bboxes_list))
			dataset = dataset.map(lambda filename, slice_num, bboxes: tuple(tf.py_function(
				func=self.input_parser, inp=[filename, slice_num, bboxes], Tout=[tf.float32,tf.float32]
				)))

		def input_parser(self, filename, slice_num, bboxes):
			print(filename, slice_num, bboxes)

			return 0,0