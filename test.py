from preprocess import *
from MainModel import *
from DataLoader import *
from tqdm import tqdm

import argparse
import pretrainedmodels
import pretrainedmodels.utils as utils

def get_data(opt, test_img_insts, test_src_insts, transform, load_img=None):
	return_data = DataLoader(
                load_img=load_img,
		transform=transform,
		image_dir=opt.image_dir,
		batch_size=opt.batch_size,
		src_word2idx=None,
		tgt_word2idx=None,
		img_insts=test_img_insts,
		src_insts=test_src_insts,
		tgt_insts=None,
		shuffle=False)

	return return_data

def test(model, opt, tgt_dict, test_img_insts, test_src_insts, transform, load_img=None):
	testing_data = get_data(opt, test_img_insts, test_src_insts, transform, load_img)
	tgt_dict = {value: key for key, value in tgt_dict.items()}
	t = tqdm(testing_data, mininterval=1, desc='Testing', leave=False)
	output_file = opt.output + ".txt"
	target = open(output_file, "wb")
	for batch in t:
		# forward
		if test_img_insts is None:
			src = batch 
			src = (src[0].cuda(), src[1].cuda())
			ini_seq = torch.ones((src[0].shape[0], 1)).cuda().long()
			ini_pos = torch.ones((src[1].shape[0], 1)).cuda().long()
			tgt = (ini_seq, ini_pos)
			for a in range(1, opt.max_len+2):
				pred = model(src=src, tgt=tgt, inference=True)
				tgt_seq = pred.argmax(2)
				tgt_seq = torch.cat((ini_seq, tgt_seq), dim=1)
				tgt_pos = torch.arange(1, a+2).cuda().long() #.repeat(tgt_seq.shape[0], 1)
				tgt = (tgt_seq, tgt_pos)
		else:
			img, src = batch
			img = img.cuda()
			src = (src[0].cuda(), src[1].cuda())
			ini_seq = torch.ones((src[0].shape[0], 1)).cuda().long()
			ini_pos = torch.ones((src[1].shape[0], 1)).cuda().long()
			tgt = (ini_seq, ini_pos)
			for a in range(1, opt.max_len):
				pred = model(img=img, src=src, tgt=tgt, inference=True)
				tgt_seq = pred.argmax(2)
				tgt_seq = torch.cat((ini_seq, tgt_seq), dim=1)
				tgt_pos = torch.arange(1, a+2).cuda().long()
				tgt = (tgt_seq, tgt_pos)

		translated = tgt_seq.tolist()
		for a in range(len(tgt_seq)):
			line = tgt_seq[a].tolist()[1:]
			if 2 in line:
				line = line[:line.index(2)]
			sentence = ""
			for b in range(len(line)):
				sentence += tgt_dict[line[b]] + " "
			sentence = sentence[:-1] + "\n"
			target.write(sentence.encode("utf-8"))
	target.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-image_dir', default=None)
	parser.add_argument('-test_img', default=None)
	parser.add_argument('-test_src', default=None)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-pretrainedmodel', default=None)
	parser.add_argument('-use_CBAM', action='store_true')
	parser.add_argument('-chkpt', required=True)
	parser.add_argument('-src_vocab', required=True)
	parser.add_argument('-tgt_vocab', required=True)
	parser.add_argument('-max_len', default=50)
	parser.add_argument('-output', default="translated")
	opt = parser.parse_args()

	# Loading dataset
	test_img_insts=None
	if opt.image_dir is not None:
		test_img_insts = read_image_instances_from_file(opt.test_img)
	test_src_word_insts = read_instances_from_file(opt.test_src, opt.max_len)

	src_dict = json.load(open(opt.src_vocab))
	tgt_dict = json.load(open(opt.tgt_vocab))

	test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, src_dict)

	# Getting train mode
	backbone=None
	mode = "MT"
	transform = None
	load_img = None
	model_size = None
	if opt.image_dir is not None:
		mode = "MMT"
		backbone = pretrainedmodels.__dict__[opt.pretrainedmodel](num_classes=1000, pretrained='imagenet')
		model_size = backbone.features(torch.zeros(backbone.input_size).unsqueeze(0)).size()[1]
		load_img = utils.LoadImage()
		transform = utils.TransformImage(backbone, scale=1.0,
			random_crop=False, random_hflip=False, random_vflip=False,
			preserve_aspect_ratio=False)

	model = MainModel(model_size, len(src_dict), len(tgt_dict), mode, backbone, opt.use_CBAM, opt.max_len+2).cuda()
	chkpt = torch.load(opt.chkpt)
	model.load_state_dict(chkpt['model'])
	model.eval()

	test(model, opt, tgt_dict, test_img_insts, test_src_insts, transform=transform, load_img=load_img)
