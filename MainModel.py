import torch.nn as nn
import torch

class ImageEncoder(nn.Module):
	def __init__(self, backbone):
		super(ImageEncoder, self).__init__()

		# backend
		modules = list(backbone.children())[:-2]
		self.backbone = nn.Sequential(*modules)

	def forward(self, img):
		out = self.backbone(img)
		return out

class ChannelAttention(nn.Module):
	def __init__(self, model_size):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1 = nn.Conv2d(model_size, model_size // 16, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(model_size // 16, model_size, 1, bias=False)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)

class MainModel(nn.Module):
	def __init__(self, model_size, n_src_vocab, n_tgt_vocab, mode, backbone, use_CBAM, max_sent_len):
		super(MainModel, self).__init__()

		self.dropout = nn.Dropout(0.1)

		if mode == "MMT":
			self.image_encoder = ImageEncoder(backbone)
			if use_CBAM:
				self.ca = ChannelAttention(model_size)
				self.ca_dropout = nn.Dropout(0.0)
			self.linear1 = nn.Linear(model_size, 512)
			self.linear1_dropout = nn.Dropout(0.0)

			self.activation = nn.ReLU()

			self.linear2 = nn.Linear(512, 512)
			self.linear2_dropout = nn.Dropout(0.0)

		# source text layers
		self.src_seq_emb = nn.Embedding(n_src_vocab, 512)
		self.src_pos_emb = nn.Embedding(max_sent_len+1, 512)
		self.src_encoder_layer = nn.TransformerEncoderLayer(512, 8)
		self.src_encoder = nn.TransformerEncoder(self.src_encoder_layer, num_layers=6)

		# target text layers
		self.tgt_seq_emb = nn.Embedding(n_tgt_vocab, 512)
		self.tgt_pos_emb = nn.Embedding(max_sent_len+1, 512)
		if mode == "MT":
			self.tgt_decoder_layer = nn.TransformerDecoderLayer(512, 8)
			self.tgt_decoder = nn.TransformerDecoder(self.tgt_decoder_layer, num_layers=6)
		else:
			self.tgt_decoder_layer = nn.MultimodalTransformerDecoderLayer(512, 8)
			self.tgt_decoder = nn.MultimodalTransformerDecoder(self.tgt_decoder_layer, num_layers=6)

		# output layer
		self.out_proj = nn.Linear(512, n_tgt_vocab)

		self.mode = mode
		self.use_CBAM = use_CBAM

	def get_trainable_parameters(self):
		if self.mode == "MMT":
			freezed_param_ids = set(map(id, self.image_encoder.parameters()))
			params = list(p for p in self.parameters() if id(p) not in freezed_param_ids)
		else:
			params = list(p for p in self.parameters())
		return params

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask.cuda()

	def get_non_pad_mask(self, seq):
		mask = seq.ne(0).type(torch.float)
		mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
		return mask.cuda()

	def forward(self, img=None, src=None, tgt=None, inference=False):
		# Encoding image
		if img is not None:
			img_mem = self.image_encoder(img)
			if self.use_CBAM:
				# Channel Attention Layer
				img_mem = self.ca_dropout(self.ca(img_mem)) * img_mem

			# Positionwise Feedforward Layer
			img_mem = img_mem.contiguous().view(img_mem.shape[0], img_mem.shape[1], -1)
			img_mem = img_mem.transpose(1, 2)
			img_mem = self.linear2_dropout(self.linear2(self.linear1_dropout(self.activation(self.linear1(img_mem)))))

		# Encoding source text
		src_seq, src_pos = src

		src_key_padding_mask = self.get_non_pad_mask(src_seq).bool()
		src_memory_key_padding_mask = self.get_non_pad_mask(src_seq).bool()

		src_seq_emb = self.src_seq_emb(src_seq)
		src_pos_emb = self.src_pos_emb(src_pos)
		src_emb = src_seq_emb + src_pos_emb
		src_emb = self.dropout(src_emb)
		src_emb = src_emb.transpose(0, 1)
		src_mem = self.src_encoder(src_emb, None, src_key_padding_mask)

	 	# Decoding target text
		tgt_seq, tgt_pos = tgt
		if not inference:
			tgt_seq = tgt_seq[:, :-1]
			tgt_pos = tgt_pos[:, :-1]

		tgt_mask = self.generate_square_subsequent_mask(tgt_seq.shape[1])
		tgt_key_padding_mask = self.get_non_pad_mask(tgt_seq).bool()

		tgt_seq_emb = self.tgt_seq_emb(tgt_seq)
		tgt_pos_emb = self.tgt_pos_emb(tgt_pos)
		tgt_emb = tgt_seq_emb + tgt_pos_emb
		tgt_emb = self.dropout(tgt_emb)
		tgt_emb = tgt_emb.transpose(0, 1)
		if img is None:
			tgt_mem = self.tgt_decoder(tgt_emb, src_mem, tgt_mask, None, tgt_key_padding_mask, src_memory_key_padding_mask)
		else:
			img_mem = img_mem.transpose(0, 1)
			tgt_mem = self.tgt_decoder(img_mem, tgt_emb, src_mem, tgt_mask, None, tgt_key_padding_mask, src_memory_key_padding_mask)
		tgt_mem = tgt_mem.transpose(0, 1)

		out_feat = self.out_proj(tgt_mem)

		return out_feat
