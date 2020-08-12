
def read_class_names(filename):
	with open(filename) as f:
		classnames = f.readlines()

	classnames = [classname.rstrip("\n") for classname in classnames]

	return classnames