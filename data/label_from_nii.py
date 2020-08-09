import os
from tqdm import tqdm, trange
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# DATA_DIR = "./data/nii/training"
# TXT_LABEL_FILENAME = "./data/nii/label_training.txt"

DATA_DIR = "./nii/training"
TXT_LABEL_FILENAME = "./nii/label_training.txt"
DIRECTION = "3D" # AXIAL, CORONAL, SAGITTAL, 3D
IMAGE_FILENAME = "image.nii"
LABEL_FILENAME = "label.nii"

# for plotting only
CLASS_NAMES = ["background", "liver", "lesion"]
PLOT = False

def bboxes_from_slice(image_slice, label_slice,plot=False):
	labelStatFilter = sitk.LabelStatisticsImageFilter()
	labelStatFilter.Execute(image_slice, label_slice)

	bboxes = []
	for label in labelStatFilter.GetLabels():
		if label == 0:
			continue

		# connected components
		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(label)
		binaryThresholdFilter.SetUpperThreshold(label)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_slice_ = binaryThresholdFilter.Execute(label_slice)

		ccFilter = sitk.ConnectedComponentImageFilter()
		label_slice_ = ccFilter.Execute(label_slice_)

		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		labelShapeFilter.Execute(label_slice_)

		for cc_region in labelShapeFilter.GetLabels():
			(x, y, w, h) = labelShapeFilter.GetBoundingBox(cc_region)
			bboxes.append((x,y,w,h,label))

	# plot to debug
	if plot:
		image_np = sitk.GetArrayFromImage(image_slice)
		image_np = image_np/1024

		# Create figure and axes
		fig,ax = plt.subplots(1)

		# Display the image
		ax.imshow(image_np,cmap="gray")
		ax.set_axis_off()

		# Create a Rectangle patch
		for (x, y, w, h, label) in bboxes:
			if label == 1:
				color = "r"
			elif label == 2:
				color = "c"
			rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor="none")
			ax.text(x,y-3, CLASS_NAMES[label], color="w")

			# Add the patch to the Axes
			ax.add_patch(rect)

		plt.show()

	return bboxes

def bbox_from_volume(image, label):
	labelStatFilter = sitk.LabelStatisticsImageFilter()
	labelStatFilter.Execute(image, label)

	bboxes = []
	for label_num in labelStatFilter.GetLabels():
		if label_num == 0:
			continue

		# connected components
		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(label_num)
		binaryThresholdFilter.SetUpperThreshold(label_num)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_ = binaryThresholdFilter.Execute(label)

		ccFilter = sitk.ConnectedComponentImageFilter()
		label_ = ccFilter.Execute(label_)

		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		labelShapeFilter.Execute(label_)

		for cc_region in labelShapeFilter.GetLabels():
			(x, y, z, w, h, d) = labelShapeFilter.GetBoundingBox(cc_region)
			bboxes.append((x,y,z,w,h,d,label_num))

	return bboxes

def generate_txt_label_2d(label_path, image,label):
	bbox_labels = []

	if DIRECTION == "AXIAL":
		pbar = tqdm(range(label.GetSize()[2]))
	elif DIRECTION == "CORONAL":
		pbar = tqdm(range(label.GetSize()[1]))
	elif DIRECTION == "SAGITTAL":
		pbar = tqdm(range(label.GetSize()[0]))

	for i in pbar:
		# check if the slice contains label
		extractor = sitk.ExtractImageFilter()

		if DIRECTION == "AXIAL":
			size = [label.GetSize()[0],label.GetSize()[1],0]
			index = [0,0,i]
		elif DIRECTION == "CORONAL":
			size = [label.GetSize()[0],0,label.GetSize()[2]]
			index = [0,i,0]
		elif DIRECTION == "SAGITTAL":
			size = [0,label.GetSize()[1],label.GetSize()[2]]
			index = [i,0,0]

		extractor.SetSize(size)
		extractor.SetIndex(index)
		image_slice = extractor.Execute(image)
		label_slice = extractor.Execute(label) 
		
		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(1)
		binaryThresholdFilter.SetUpperThreshold(255)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_slice_ = binaryThresholdFilter.Execute(label_slice)

		statFilter = sitk.StatisticsImageFilter()
		statFilter.Execute(label_slice_)

		bbox_txt = os.path.abspath(label_path) + " " + str(i)
		if statFilter.GetSum() > 1:
			bboxes = bboxes_from_slice(image_slice,label_slice,plot=PLOT)
				
			for (x, y, w, h, class_num) in bboxes:
				bbox_txt = "{} {},{},{},{},{}".format(bbox_txt,x,y,w,h,class_num)
		
		bbox_labels.append(bbox_txt)

	return bbox_labels

def generate_txt_label_3d(label_path, image, label):
	# check if label contain things 
	binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
	binaryThresholdFilter.SetLowerThreshold(1)
	binaryThresholdFilter.SetUpperThreshold(255)
	binaryThresholdFilter.SetInsideValue(1)
	binaryThresholdFilter.SetOutsideValue(0)
	label_ = binaryThresholdFilter.Execute(label)

	statFilter = sitk.StatisticsImageFilter()
	statFilter.Execute(label_)

	bbox_txt = os.path.abspath(label_path) + " "
	if statFilter.GetSum() > 1:
		bboxes = bbox_from_volume(image,label)
			
		for (x, y, z, w, h, d, class_num) in bboxes:
			bbox_txt = "{} {},{},{},{},{},{},{}".format(bbox_txt,x,y,z,w,h,d,class_num)

	return [bbox_txt]

def append_to_txt_label(image_path, label_path):
	# check existence of required files
	if not (os.path.exists(image_path) and os.path.exists(label_path)):
		return

	reader = sitk.ImageFileReader()
	reader.SetFileName(image_path)
	image = reader.Execute()

	reader.SetFileName(label_path)
	label = reader.Execute()

	# check image size consist
	if not (image.GetSize() == label.GetSize() and image.GetDirection() == label.GetDirection() and image.GetSpacing() == label.GetSpacing()):
		return

	# create label output 
	output_file = open(TXT_LABEL_FILENAME, "w")

	if DIRECTION == "AXIAL" or DIRECTION == "CORONAL" or DIRECTION == "SAGITTAL":
		# 2D model
		label_txt = generate_txt_label_2d(label_path, image, label)
	elif DIRECTION == "3D":
		# 3D model
		label_txt = generate_txt_label_3d(label_path, image, label)
	else:
		print("DIRECTION should be AXIAL, CORONAL, SAGITTAL or 3D")

	output_file.write("\n".join(label_txt)+"\n")
	output_file.close()


def main():
	pbar = tqdm(os.listdir(DATA_DIR))
	for case in pbar:
		append_to_txt_label(os.path.join(DATA_DIR,case,IMAGE_FILENAME), os.path.join(DATA_DIR,case,LABEL_FILENAME))

if __name__ == "__main__":
	main()