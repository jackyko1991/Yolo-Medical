# Prepare Data

## Medical Image Data Format
In general medical images are stored as DICOMs which contains patient and scanning information. For convenience segmentation data will be stored as Analyze/ NIFTI format which stacks multiple slices into single file. In this repository we will first focus on NIFTI image input and perform detection in 2D

## Labels
Classical Yolo train with label file as following:

- dataset.txt
```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

- classes.txt
```
person
bicycle
car
...
toothbrush
```

### Data Folder Hierarchy
Place the data in an arbitrary folder `<data_dir>` with follow folder hierarchy:

#### NIFTI
    <data_dir>                # All data
    ├── testing               # Put all testing data here
    |   ├── case1            
    |   |   ├── img.nii.gz    # Image for testing
    |   |   └── label.nii.gz  # Corresponding label for testing
    |   ├── case2
    |   ├──...
    ├── training              # Put all training data here
    |   ├── case1             # foldername for the cases is arbitrary
    |   |   ├── img.nii.gz    # Image for training
    |   |   └── label.nii.gz  # Corresponding label for training
    |   ├── case2
    |   ├──...
    ├── evaluation            # Put all evaluation data here
    |   ├── img1.nii.gz       # Image for evaluation, filename is arbitrary
    |   ├── img2.nii.gz
    |   ├──...
    ├── evaluation_plot       # Evaluated result plot will be here, only for 2D inference
    |   ├── img1.nii.gz       # Image for evaluation
    |   |   ├── slice000.png
    |   |   ├── slice001.png
    |   |   ├──...
    |   ├── img2.nii.gz
    |   ├──...
    ├── label_testing.txt     # Text labels
    ├── label_training.txt
    └── label_evaluation.txt

### Labels from segmentation 
Here we provide a tool to convert segmentation label from NIFTI file to a slightly modified format in specific view direction:

- dataset_axial.txt 
```
xxx/xxx.nii.gz 0 18.19,6.32,424.13,421.83,2 323.86,2.65,640.0,421.94,2 
xxx/xxx.nii.gz 1 48,240,195,371,11 8,12,352,498,14
# image_path slice x_min, y_min, x_max, y_max, class_id  x_min, y_min, ..., class_id 
# make sure that x_max < width and y_max < height
```

Or we may export the label as 3D bounding box:

- dataset_3D.txt 
```
xxx/xxx.nii.gz 18.19,6.32,5.24,424.13,421.83,410.2,2 323.86,2.65,128.2,640.0,421.94,152.4,2 
xxx/xxx.nii.gz 48,240,20.4,195,371,120.8,0 8,12,64.4,352,498,160.2,1
# image_path x_min, y_min, z_min, x_max, y_max, z_max, class_id  x_min, y_min, z_min, ..., class_id 
# make sure that x_max < width and y_max < height
```

Edit the following lines in `./data/label_from_nii.py` to fit your folder path

Execute the file from repository root with

```bash
python ./data/label_from_nii.py
```

**Note: The bounding box is in pixel coordinate, spacial resolution is not taken into consideration for this model**

### Create your own dataset from scratch
**To be developed**

#### DICOM
#### NIFTI

## Usage