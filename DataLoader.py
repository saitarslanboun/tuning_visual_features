from torch.autograd import Variable

import numpy as np
import random
import torch

''' Data Loader class for training iteration '''

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, load_img, transform, image_dir, src_word2idx, tgt_word2idx,
            img_insts=None, src_insts=None, tgt_insts=None,
            batch_size=64, shuffle=True, test=False):

        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._img_insts = img_insts
        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        self._src_word2idx = src_word2idx
        self._tgt_word2idx = tgt_word2idx

        self._iter_count = 0

        self._need_shuffle = shuffle

        self.image_dir = image_dir
        self.transform = transform

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._tgt_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts:
            if self.transform is None:
                paired_insts = list(zip(self._src_insts, self._tgt_insts))
                random.shuffle(paired_insts)
                self._src_insts, self._tgt_insts = zip(*paired_insts)
            else:
                paired_insts = list(zip(self._img_insts, self._src_insts, self._tgt_insts))
                random.shuffle(paired_insts)
                self._img_insts, self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            if self.transform is None:
                paired_insts = list(self._src_insts)
            else:
                paired_insts = list(zip(self._img_insts, self._src_insts))
                self._img_insts, self._src_insts = zip(*paired_insts)
            random.shuffle(paired_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)
  
            inst_data = np.array([
                inst + [0] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            inst_position_tensor = Variable(
                torch.LongTensor(inst_position), volatile=self.test)

            inst_data_tensor = inst_data_tensor.cuda()
            inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if self.transform is not None:
                img_insts = self._img_insts[start_idx:end_idx]
                image_data = []
                for a in range(len(img_insts)):
                    img_inst = img_insts[a]
                    try:
                        inst = os.path.join(self.image_dir, img_inst)
                        image = load_img(inst)
                        image = self.transform(image)
                    except:
                        image = torch.zeros((3, 224, 224))
                    image_data.append(image)

                image_data = torch.stack(image_data).cuda()
            src_insts = self._src_insts[start_idx:end_idx]
            src_data, src_pos = pad_to_longest(src_insts)

            if not self._tgt_insts:
                if self.transform is None:
                    return (src_data, src_pos)
                else:
                    return image_data, (src_data, src_pos)
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)

                if self.transform is None:
                    return (src_data, src_pos), (tgt_data, tgt_pos)
                else:
                    return image_data, (src_data, src_pos), (tgt_data, tgt_pos)

        else:
            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
