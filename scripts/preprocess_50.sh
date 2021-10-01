python ../Model/preprocess.py                    \
	-train_img ../dataset/train.img	    \
        -train_src ../dataset/train.en           \
        -train_tgt ../dataset/train.de           \
	-valid_img ../dataset/val.img            \
        -valid_src ../dataset/val.en             \
        -valid_tgt ../dataset/val.de             \
        -src_vocab ../dataset/train.en.json	\
        -tgt_vocab ../dataset/train.de.json	\
        -save_data ../dataset/data.pt
