import numpy as np
import random

class Vocabulary:
    def __init__ (self):
        self._w2i = {}
        self._i2w = []
        self.add_all()

    def add_all(self):
        pos = 0
        with open('data/word_list.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                self._w2i[line] = pos
                self._i2w.append(line)
                pos += 1

    def w2i(self, word):
        if word in self._w2i:
            return self._w2i[word]
        else:
            return self._w2i['unk']

    def i2w(self, i):
        return self._i2w[i]


class figer_data_single_label:
    # def __init__ (self, batch_size = 1000, entity_file = 'data/state_of_the_art_train_word_with_context.txt', \
    #                 context_file = 'data/state_of_the_art_train_tagged_context.txt', \
    #                 type_file = 'data/state_of_the_art_train_Types_with_context.npy', negative_sampling_rate = 0.2):
    def __init__ (self, entity_file, \
                context_file, type_file, batch_size = 1000, negative_sampling_rate = 1.0):
        self.load_data(entity_file, context_file, type_file)
        self.train_pos = 0
        self.validation_pos = 0
        self.test_pos = 0
        self.batch_size = min(batch_size, len(self.Words))
        self.total_batch_num = int(len(self.Words) / self.batch_size)
        self.vob = Vocabulary()
        self.negative_sampling_rate = negative_sampling_rate


    def load_data(self, entity_file, context_file, type_file):
        with open('all_types.txt', 'r') as f:
            self.all_types = [data.replace('\n','') for data in f]
        with open('held_out_types.txt', 'r') as f:
            self.held_out_types = [data.replace('\n','') for data in f]
        # with open('data/labelid2emb.pkl', 'r') as f:
        #     self.label_id2emb = pickle.load(f)

        self.Words = []
        with open(entity_file, 'r') as f:
            for line in f:
                self.Words.append(line.replace('\n',''))

        self.Contexts = []
        with open(context_file, 'r') as f:
            for line in f:
                self.Contexts.append(line.replace('\n',''))

        self.Types = np.load(type_file)
        self.r = range(0, len(self.Words))

    def shuffle(self):
        random.shuffle(self.r)


    def gen_random_pos_emb(self, pos_emb_len = 50):
        self.static_pos_emb = {}
        for i in range(-100, 100):
            self.static_pos_emb[i] = np.random.random(pos_emb_len)


    def next_batch(self, window_size = 10, select_lable_ids = range(0,113)):
        ret_Entity_var_ids = []
        ret_r_Left_context_ids = []
        ret_Right_context_ids = []
        ret_Label_ids = []
        ret_Ys = []

        count = 0
        while count < self.batch_size:
            local_Entity_var_ids = []
            local_Left_context_ids = []
            local_Right_context_ids = []

            for e in self.Words[self.r[self.train_pos]].split():
                local_Entity_var_ids.append(self.vob.w2i(e))

            local_r_Left_context_ids, local_Right_context_ids = \
                    self.get_r_left_right_context(self.Contexts[self.r[self.train_pos]], window_size)
            raw_Ys = self.Types[self.r[self.train_pos]]

            for label_id in range(0, 113):
                if not label_id in select_lable_ids:
                    continue
                if raw_Ys[label_id] == 1.0:
                    ret_Entity_var_ids.append(local_Entity_var_ids)
                    ret_r_Left_context_ids.append(local_r_Left_context_ids)
                    ret_Right_context_ids.append(local_Right_context_ids)
                    ret_Label_ids.append(label_id)
                    ret_Ys.append([1.0])
                else:
                    if np.random.uniform() < self.negative_sampling_rate:
                        ret_Entity_var_ids.append(local_Entity_var_ids)
                        ret_r_Left_context_ids.append(local_r_Left_context_ids)
                        ret_Right_context_ids.append(local_Right_context_ids)
                        ret_Label_ids.append(label_id)
                        ret_Ys.append([0.0])

            self.train_pos = (self.train_pos + 1) % (len(self.Words))
            count += 1

        ret_Entity_ids, ret_Entity_lens = vstack_list_padding_2d(ret_Entity_var_ids, padding_element = self.vob.w2i('_my_null_'))

        ret_r_Left_context_ids, ret_Left_context_lens = vstack_list_padding_2d(ret_r_Left_context_ids, padding_element = self.vob.w2i('_my_null_'))
        ret_Right_context_ids, ret_Right_context_lens = vstack_list_padding_2d(ret_Right_context_ids, padding_element = self.vob.w2i('_my_null_'))

        ret_Left_context_ids = []

        for e in ret_r_Left_context_ids:
            ret_Left_context_ids.append(e[::-1])

        ret_Entity_ids = np.asarray(ret_Entity_ids, dtype=np.float32)
        ret_Entity_lens = np.asarray(ret_Entity_lens, dtype=np.float32)

        ret_Left_context_ids = np.asarray(ret_Left_context_ids, dtype=np.float32)
        ret_Left_context_lens = np.asarray(ret_Left_context_lens, dtype=np.float32)

        ret_Right_context_ids = np.asarray(ret_Right_context_ids, dtype=np.float32)
        ret_Right_context_lens = np.asarray(ret_Right_context_lens, dtype=np.float32)

        ret_Label_ids = np.asarray(ret_Label_ids, dtype=np.float32)

        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)

        return ret_Entity_ids, ret_Entity_lens, ret_Left_context_ids, \
                ret_Left_context_lens, ret_Right_context_ids, \
                ret_Right_context_lens, ret_Label_ids, ret_Ys


    def get_r_left_right_context(self, context_string, window_size):
        words = context_string.split()

        l_pos = len(words)
        r_pos = len(words)
        for i in range(0, len(words)):
            if words[i] == '<e>':
                l_pos = i
            if words[i] == '</e>':
                r_pos = i

        # r_l_context: r_ for reverse
        r_l_context = []
        for i in range(l_pos-1, -1, -1):
            r_l_context.append(self.vob.w2i(words[i]))
            if len(r_l_context) >= window_size:
                break

        r_context = []

        for i in range(r_pos+1, len(words)):
            r_context.append(self.vob.w2i(words[i]))
            if len(r_context) >= window_size:
                break


        return r_l_context, r_context

def vstack_list_padding_2d(data, padding_element = 0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.zeros((len(data), max_len), dtype=dtype)
    arr.fill(padding_element)
    for i, row in enumerate(data):
        arr[i, 0:len(row)] = row
    return arr, np.array(lengths, dtype=np.int32)
