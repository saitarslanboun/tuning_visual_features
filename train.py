from MainModel import *
from ScheduledOptim import *
from tqdm import tqdm
from DataLoader import *
from nltk.translate.gleu_score import corpus_gleu

import pretrainedmodels
import pretrainedmodels.utils as utils
import argparse
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import codecs
import math
import os

def get_data(data, ind, split, opt, transform, load_img):
	if split == "train":
		data_pair = list(zip(*data['train']['pair'][ind]))
		shuffle=True
		batch_size = opt.batch_size
	elif split == "valid":
		data_pair = list(zip(*data['valid']['pair']))
		shuffle=False
		batch_size = opt.batch_size
	else:
		data_pair = list(zip(*data['valid']['pair']))
		shuffle=False
		batch_size=opt.batch_size

	if transform is None:
		image_dir = None
		img_insts = None
		src_insts = data_pair[0]
		tgt_insts = data_pair[1]
	else:
		image_dir = opt.image_dir
		img_insts = data_pair[0]
		src_insts = data_pair[1]
		tgt_insts = data_pair[2]

	return_data = DataLoader(
                load_img=load_img,
		transform=transform,
		image_dir=opt.image_dir,
		batch_size=batch_size,
		src_word2idx=data['dict']['src'],
		tgt_word2idx=data['dict']['tgt'],
		img_insts=img_insts,
		src_insts=src_insts,
		tgt_insts=tgt_insts,
		shuffle=shuffle)

	return return_data

def calc_loss(pred, gold):
	gold = gold.contiguous().view(-1)

	eps = 0.1
	n_class = pred.size(1)

	one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
	one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
	log_prb = F.log_softmax(pred, dim=1)

	non_pad_mask = gold.ne(0)
	loss = -(one_hot * log_prb).sum(dim=1)
	loss = loss.masked_select(non_pad_mask).sum()

	return loss

def calc_performance(pred, gold):
	pred = pred.contiguous().view(-1, pred.shape[2])
	loss = calc_loss(pred, gold)

	pred = pred.max(1)[1]
	gold = gold.contiguous().view(-1)
	non_pad_mask = gold.ne(0)
	n_correct = pred.eq(gold)
	n_correct = n_correct.masked_select(non_pad_mask).sum().item()

	return loss, n_correct

def valid_update(model, validation_data):
	model.eval()

	n_total_words = 0
	n_total_correct = 0
	total_loss = 0
	total_acc = 0
	cntr = 0

	t = tqdm(validation_data, mininterval=1, desc='Validation', leave=False)
	for batch in t:

		# prepare data
		gold = batch[-1][0][:, 1:].cuda()

		# forward
		if len(batch) == 2:
			src, tgt = batch
			src = (src[0].cuda(), src[1].cuda())
			tgt = (tgt[0].cuda(), tgt[1].cuda())
			pred = model(src=src, tgt=tgt)
		else:
			img, src, tgt = batch
			img = img.cuda()
			src = (src[0].cuda(), src[1].cuda())
			tgt = (tgt[0].cuda(), tgt[1].cuda())
			pred = model(img=img, src=src, tgt=tgt)

		# backward
		loss, n_correct = calc_performance(pred, gold)

		# note keeping
		n_words = gold.ne(0).sum().item()
		n_total_words += n_words
		n_total_correct += n_correct
		total_loss += loss.item()
		acc = (100.0 * n_correct) / n_words
		total_acc += acc
		cntr += 1
		loss_i = total_loss / cntr
		acc_i = total_acc / cntr		

		description = "Loss: " + str(("%.2f" % loss_i)) + " Acc: " + str(("%.2f" % acc_i))
		t.set_description(description)

	return total_loss/float(n_total_words), float(n_total_correct)/float(n_total_words)
	
def train_update(model, training_data, optimizer):
	model.train()

	n_total_words = 0
	n_total_correct = 0
	total_loss = 0
	total_acc = 0
	cntr = 0

	t = tqdm(training_data, mininterval=1, desc='Training', leave=False)
	for batch in t:

		# prepare data
		gold = batch[-1][0][:, 1:].cuda()

		# forward
		optimizer.zero_grad()
		if len(batch) == 2:
			src, tgt = batch
			src = (src[0].cuda(), src[1].cuda())
			tgt = (tgt[0].cuda(), tgt[1].cuda())
			pred = model(src=src, tgt=tgt)
		else:
			img, src, tgt = batch
			img = img.cuda()
			src = (src[0].cuda(), src[1].cuda())
			tgt = (tgt[0].cuda(), tgt[1].cuda())
			pred = model(img=img, src=src, tgt=tgt)

		# backward
		loss, n_correct = calc_performance(pred, gold)
		loss.backward()

		# update parameters
		optimizer.step_and_update_lr()

		# note keeping
		n_words = gold.ne(0).sum().item()
		n_total_words += n_words
		n_total_correct += n_correct
		total_loss += loss.item()
		acc = (100.0 * n_correct) / n_words
		total_acc += acc
		cntr += 1
		loss_i = total_loss / cntr
		acc_i = total_acc / cntr

		description = "Loss: " + str(("%.2f" % loss_i)) + " Acc: " + str(("%.2f" % acc_i))
		t.set_description(description)

	return total_loss/float(n_total_words), float(n_total_correct)/float(n_total_words)

def test_update(model, validation_data, max_sent_le, tgt_dict, reference):

	tgt_dict = {value: key for key, value in tgt_dict.items()}

	t = tqdm(validation_data, mininterval=1, desc='Testing', leave=False)
	target = open("tmp", "wb")
	for batch in t:

		# forward
		if len(batch) == 2:
			src, rtgt = batch 
			src = (src[0].cuda(), src[1].cuda())
			ini_seq = torch.ones((src[0].shape[0], 1)).cuda().long()
			ini_pos = torch.ones((src[1].shape[0], 1)).cuda().long()
			tgt = (ini_seq, ini_pos)
			for a in range(1, max_sent_len):
				pred = model(src=src, tgt=tgt, inference=True)
				tgt_seq = pred.argmax(2)
				tgt_seq = torch.cat((ini_seq, tgt_seq), dim=1)
				tgt_pos = torch.arange(1, a+2).cuda().long().repeat(tgt_seq.shape[0], 1)
				tgt = (tgt_seq, tgt_pos)
		else:
			img, src, rtgt = batch
			img = img.cuda()
			src = (src[0].cuda(), src[1].cuda())
			ini_seq = torch.ones((src[0].shape[0], 1)).cuda().long()
			ini_pos = torch.ones((src[1].shape[0], 1)).cuda().long()
			tgt = (ini_seq, ini_pos)
			for a in range(1, max_sent_len):
				pred = model(img=img, src=src, tgt=tgt, inference=True)
				tgt_seq = pred.argmax(2)
				tgt_seq = torch.cat((ini_seq, tgt_seq), dim=1)
				tgt_pos = torch.arange(1, a+2).cuda().long()
				tgt = (tgt_seq, tgt_pos)

		#translated = tgt_seq.tolist()
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

	translation_file = codecs.open("tmp", encoding="utf-8").readlines()
	translation = []
	for a in range(len(translation_file)):
		line = translation_file[a].strip().split(' ')
		translation.append(line)

	score = corpus_gleu(reference, translation)		
	return score

def train(model, num_shards, optimizer, opt, data, max_sent_len, tgt_dict, transform, load_img=None):
	log_train_file = None
	log_valid_file = None

	ref_file = codecs.open(opt.val_reference, encoding="utf-8").readlines()
	reference = []
	for a in range(len(ref_file)):
		line = ref_file[a].strip().split(' ')
		reference.append([line])

	if opt.log:
		log_train_file = opt.log + "_train_log"
		log_valid_file = opt.log + "_valid_log"
		log_test_file = opt.log + "_test_log"

		print('[Info] Training performance will be written to file: {}'.format(log_train_file))
		print('[Info] Validation performance will be written to file: {}'.format(log_valid_file))
		print('[Info] Testing performance will be written to file: {}'.format(log_test_file))

		with open(log_train_file, 'w') as log_trf:
			log_trf.write('update,loss,ppl,accuracy\n')
		with open(log_valid_file, 'w') as log_vf:
			log_vf.write('update,loss,ppl,accuracy\n')
		with open(log_test_file, 'w') as log_tf:
			log_tf.write('update,bleu\n')

	cntr_update = 0
	bleus = []
	tol_cntr = 0
	while(True):
		for update_i in range(num_shards):
			print('[Update', cntr_update, ']')

			training_data = get_data(data, update_i, "train", opt, transform, load_img)
			validation_data = get_data(data, update_i, "valid", opt, transform, load_img)
			testing_data = get_data(data, update_i, "test", opt, transform, load_img)

			train_loss, train_accu = train_update(model, training_data, optimizer)
			valid_loss, valid_accu = valid_update(model, validation_data)
			bleu_score = test_update(model, testing_data, max_sent_len, tgt_dict, reference)
			bleus.append(bleu_score)

			if bleu_score != max(bleus):
				tol_cntr += 1
				if tol_cntr == 20:
					print("No improvement for the last 20 updates!")
					exit()
			else:
				tol_cntr = 0

				model_state_dict = model.state_dict()
				checkpoint = {
					"model": model_state_dict,
					"settings": opt,
					"update": cntr_update
				}
				model_name = os.path.join(opt.save_model, "model."+str(cntr_update)+".chkpt")
				torch.save(checkpoint, model_name)

			if log_train_file:
				with open(log_train_file, 'a') as log_trf:
					log_trf.write('{update},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
						update=cntr_update, loss=train_loss,
						ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
				with open(log_valid_file, 'a') as log_vf:
					log_vf.write('{update},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
						update=cntr_update, loss=valid_loss,
						ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
				with open(log_test_file, 'a') as log_tf:
					log_tf.write('{update},{bleu}\n'.format(
						update=cntr_update, bleu=bleu_score))

		cntr_update += 1
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data', required=True)
	parser.add_argument('-val_reference', required=True)
	parser.add_argument('-image_dir', default=None)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-n_warmup_steps', type=int, default=4000)
	parser.add_argument('-log', default=None)
	parser.add_argument('-save_model', default=None)
	parser.add_argument('-pretrainedmodel', default=None)
	parser.add_argument('-use_CBAM', action='store_true')
	opt = parser.parse_args()

	# Loading dataset
	data = torch.load(opt.data)
	max_sent_len = data["settings"].max_len

	# Getting train mode
	mode = data["mode"]
	backbone=None
	transform = None
	load_img = None
	model_size = None       
	if mode == "MMT":
		backbone = pretrainedmodels.__dict__[opt.pretrainedmodel](num_classes=1000, pretrained='imagenet')
		model_size = backbone.features(torch.zeros(backbone.input_size).unsqueeze(0)).size()[1]
		load_img = utils.LoadImage()
		transform = utils.TransformImage(backbone, scale=1.0,
			random_crop=False, random_hflip=False, random_vflip=False,
			preserve_aspect_ratio=False)

	src_dict = data['dict']['src']
	tgt_dict = data['dict']['tgt']
	num_shards = data['train']['length']

	model = MainModel(model_size, len(src_dict), len(tgt_dict), mode, backbone, opt.use_CBAM, max_sent_len)

	optimizer = ScheduledOptim(
		optim.Adam(
			model.get_trainable_parameters(),
			betas=(0.9, 0.98), eps=1e-09),
		512, opt.n_warmup_steps, 0)

	model = model.cuda()

	train(model, num_shards, optimizer, opt, data, max_sent_len, tgt_dict, transform=transform, load_img=load_img)
