python ../Model/train.py				\
	-data ../dataset/data.pt			\
	-val_reference ../dataset/val.de		\
	-image_dir ../../../Flickr30kDataset/images/	\
	-pretrainedmodel resnet50			\
	-log log					\
	-save_model models
