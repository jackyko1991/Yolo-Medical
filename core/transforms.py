from core import NiftiDataset2D
from core import NiftiDataset3D

def train_transform_2d(patch_shape):
	transform =[
		# NiftiDataset2D.ManualNormalization(0,300)
		# NiftiDataset2D.Resample()
	]
	return transform

test_transform_2d =[
	
]

train_transform_3d =[
	
]

test_transform_3d =[
	
]