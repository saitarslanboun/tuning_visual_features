from mosestokenizer import *

import argparse
import codecs
import json
import torch

def read_image_instances_from_file(inst_file):
	image_insts = []
	f = codecs.open(inst_file, encoding="utf-8").readlines()
	for sent in f:
		image_insts += [sent.strip()]

	print('[Info] Get {} instances from {}.'.format(len(image_insts), inst_file))

	return image_insts

def read_instances_from_file(inst_file, max_sent_len):
	word_insts = []
	trimmed_sent_count = 0
	with open(inst_file) as f:
		for sent in f:
			words = sent.strip().split(' ')
			if len(words) > max_sent_len:
				trimmed_sent_count += 1
			word_inst = words[:max_sent_len]

			if word_inst:
				word_insts += [['<BOS>'] + word_inst + ['<EOS>']]
			else:
				word_insts += [None]

	print('[Info] Get {} instances from {}.'.format(len(word_insts), inst_file))

	if trimmed_sent_count > 0:
		print('[Warning] {} instances are trimmed to the max sentence length {}.'
			.format(trimmed_sent_count, max_sent_len))

	return word_insts

def convert_instance_to_idx_seq(word_insts, word2idx):
	idx_seqs = []
	for inst in word_insts:
		idx_seq = []
		for token in inst:
			try:
				idx_seq.append(word2idx[token])
			except:
				idx_seq.append(word2idx['<UNK>'])
		idx_seqs.append(idx_seq)
	return idx_seqs
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-train_img', default=None)
	parser.add_argument('-train_src', default=None)
	parser.add_argument('-train_tgt', default=None)
	parser.add_argument('-valid_img', default=None)
	parser.add_argument('-valid_src', default=None)
	parser.add_argument('-valid_tgt', default=None)
	parser.add_argument('-src_vocab', default=None)
	parser.add_argument('-tgt_vocab', default=None)
	parser.add_argument('-max_len', type=int, default=50)
	parser.add_argument('-shard_size', type=int, default=30000)
	parser.add_argument('-save_data', required=True)
	opt = parser.parse_args()
	opt.max_len = opt.max_len + 2

	# Training set
	if opt.train_img is not None:
		train_img_insts = read_image_instances_from_file(opt.train_img)
	train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_len)
	train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_len)

	# Validation set
	if opt.train_img is not None:
		valid_img_insts = read_image_instances_from_file(opt.valid_img)
	valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_len)
	valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_len)

	# Load Vocabulary
	src_word2idx = json.load(open(opt.src_vocab))
	tgt_word2idx = json.load(open(opt.tgt_vocab))

	# Word to Index
	print('[Info] Convert source word instances into sequences of word index.')
	train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
	valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

	print('[Info] Convert target word instances into sequences of word index.')
	train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
	valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

	if opt.train_img is None:
		whole = list(zip(train_src_insts, train_tgt_insts))
	else:
		whole = list(zip(train_img_insts, train_src_insts, train_tgt_insts))
	whole_train_insts = [whole[i * opt.shard_size:(i + 1) * opt.shard_size] for i in range((len(whole) + opt.shard_size - 1) // opt.shard_size)]
	if opt.train_img is None:
		whole_valid_insts = list(zip(valid_src_insts, valid_tgt_insts))
	else:
		whole_valid_insts = list(zip(valid_img_insts, valid_src_insts, valid_tgt_insts))
	

	if opt.train_img is None:
		train_mode = "MT"
	else:
		train_mode = "MMT"

	data = {
		'settings': opt,
		'mode': train_mode,
		'dict':
			{
				'src': src_word2idx,
				'tgt': tgt_word2idx},
		'train':
			{
				'pair': whole_train_insts,
				'length': len(whole_train_insts)},
		'valid': {'pair': whole_valid_insts}
	}

	print('[Info] Dumping the processed data to pickle file', opt.save_data)
	torch.save(data, opt.save_data)
	print('[Info] Finish.')
