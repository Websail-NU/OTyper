import numpy as np
import random
import pickle
import gensim


class Vocabulary:
    def __init__ (self):
        self._w2i = {}
        self._i2w = []
        self.add_all()

    def add_all(self):
        pos = 0
        with open('UMLS_data/word_list.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                self._w2i[line] = pos
                self._i2w.append(line)
                pos += 1

    def w2i(self, word):
        if word in self._w2i:
            return self._w2i[word]
        elif word.lower() in self._w2i:
            return self._w2i[word.lower()]
        else:
            return self._w2i['unk']

    def i2w(self, i):
        return self._i2w[i]


class umls_data_multi_label:
    def __init__ (self, batch_size = 100, entity_file = 'UMLS_data/refined_umls_word.txt', \
                    context_file = 'UMLS_data/refined_umls_tagged_context.txt', \
                    entity_type_exact_feature_file = 'UMLS_data/umls_exact_et_features.npy', \
                    type_file = 'UMLS_data/umls_Types_with_context.npy'):
        self.shuffle_flag = 0
        self.vob = Vocabulary()
        self.load_data(entity_file, context_file, type_file, entity_type_exact_feature_file)
        self.train_pos = 0
        self.batch_size = min(batch_size, len(self.Entity_var_ids))
        self.total_batch_num = int(len(self.Entity_var_ids) / self.batch_size)



    def load_data(self, entity_file, context_file, type_file, entity_type_exact_feature_file, window_size = 10):
        self.Entity_var_ids = []
        with open(entity_file, 'r') as f:
            for line in f:
                t_l = []
                for w in line.replace('\n','').split():
                    t_l.append(self.vob.w2i(w))
                self.Entity_var_ids.append(t_l)

        self.r_Left_context_ids = []
        self.Right_context_ids = []
        with open(context_file, 'r') as f:
            for line in f:
                context = line.replace('\n','')
                t_r_Left_context_ids, t_Right_context_ids = \
                    self.get_r_left_right_context(context, window_size)
                self.r_Left_context_ids.append(t_r_Left_context_ids)
                self.Right_context_ids.append(t_Right_context_ids)

        self.Types = np.load(type_file)

        self.Exact_entity_type_features = np.load(entity_type_exact_feature_file)

        self.r = list(range(0, len(self.Entity_var_ids)))

        if self.shuffle_flag == 1:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.r)


    # select_flag == 0: select_all
    # select_flag == 1: select train label, delete ids
    # select_flag == 2: select test label, take ids

    # def next_batch(self, window_size = 10, select_flag = 0, seen_label_ids = None, unseen_label_ids = None):
    def next_batch(self, select_label_ids):
        ret_Entity_var_ids = []
        ret_r_Left_context_ids = []
        ret_Right_context_ids = []
        ret_Exact_entity_type_features = []

        ret_Ys = []


        count = 0
        while count < self.batch_size:

            local_Entity_var_ids = []
            local_Left_context_ids = []
            local_Right_context_ids = []

            local_Entity_var_ids = self.Entity_var_ids[self.r[self.train_pos]]
            local_r_Left_context_ids = self.r_Left_context_ids[self.r[self.train_pos]]
            local_Right_context_ids = self.Right_context_ids[self.r[self.train_pos]]

            local_Exact_entity_type_features = self.Exact_entity_type_features[self.r[self.train_pos]]

            local_Ys = self.Types[self.r[self.train_pos]]

            ret_Entity_var_ids.append(local_Entity_var_ids)
            ret_r_Left_context_ids.append(local_r_Left_context_ids)
            ret_Right_context_ids.append(local_Right_context_ids)
            ret_Exact_entity_type_features.append(local_Exact_entity_type_features)

            ret_Ys.append(local_Ys)

            self.train_pos = (self.train_pos + 1) % (len(self.Entity_var_ids))
            count += 1


        ret_Entity_ids, ret_Entity_lens = vstack_list_padding_2d(ret_Entity_var_ids, padding_element = self.vob.w2i('_my_null_'))

        ret_r_Left_context_ids, ret_Left_context_lens = vstack_list_padding_2d(ret_r_Left_context_ids, padding_element = self.vob.w2i('_my_null_'))
        ret_Right_context_ids, ret_Right_context_lens = vstack_list_padding_2d(ret_Right_context_ids, padding_element = self.vob.w2i('_my_null_'))

        ret_Left_context_ids = []

        for e in ret_r_Left_context_ids:
            ret_Left_context_ids.append(e[::-1])

        ret_Entity_ids = np.asarray(ret_Entity_ids, dtype=np.float32)
        ret_Entity_lens = np.asarray(ret_Entity_lens, dtype=np.float32)

        batch_size = ret_Entity_ids.shape[0]
        type_size = len(select_label_ids)

        ret_Left_context_ids = np.asarray(ret_Left_context_ids, dtype=np.float32)
        ret_Left_context_lens = np.asarray(ret_Left_context_lens, dtype=np.float32)

        ret_Right_context_ids = np.asarray(ret_Right_context_ids, dtype=np.float32)
        ret_Right_context_lens = np.asarray(ret_Right_context_lens, dtype=np.float32)

        ret_Feature_ids = np.zeros((batch_size, 70))
        t_type_only_features = np.zeros((batch_size, type_size, 3))


        ret_Entity_type_features = np.zeros((batch_size, type_size, 3))
        ret_Exact_entity_type_features = np.take(ret_Exact_entity_type_features, select_label_ids, 1)
        ret_Type_only_features = np.zeros((batch_size, type_size, 3))

        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        ret_Ys = np.take(ret_Ys, select_label_ids, 1)


        return ret_Entity_ids, ret_Entity_lens, ret_Left_context_ids, ret_Left_context_lens, \
                ret_Right_context_ids, ret_Right_context_lens, ret_Feature_ids, \
                ret_Entity_type_features, ret_Exact_entity_type_features, \
                ret_Type_only_features, ret_Ys


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


def get_train_label_emb(label_embs, Ys):
    return label_embs, Ys

def get_test_label_emb(label_embs, Ys):
    return label_embs, Ys

def vstack_list_padding_2d(data, padding_element = 0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.zeros((len(data), max_len), dtype=dtype)
    arr.fill(padding_element)
    for i, row in enumerate(data):
        arr[i, 0:len(row)] = row
    return arr, np.array(lengths, dtype=np.int32)


def get_an_example():
    a = umls_data_multi_label()
    ret = a.next_batch(list(range(0, 1387)))

    print(ret[-3].shape)


if __name__ == "__main__":
    # get_an_example()
    temp()
