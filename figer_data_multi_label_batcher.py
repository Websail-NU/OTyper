import numpy as np
import random
from sklearn.externals import joblib


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


class figer_data_multi_label:
    def __init__ (self, batch_size = 1000, entity_file = 'data/state_of_the_art_train_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_train_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_train_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_train_et_features.npy', \
                    entity_type_exact_feature_file = 'data/state_of_the_art_train_exact_et_features.npy', \
                    type_file = 'data/state_of_the_art_train_Types_with_context.npy'):
        self.shuffle_flag = 0
        self.vob = Vocabulary()
        self.load_data(entity_file, context_file, feature_file, entity_type_feature_file, entity_type_exact_feature_file, type_file)
        self.train_pos = 0
        self.validation_pos = 0
        self.test_pos = 0
        self.batch_size = min(batch_size, len(self.Entity_var_ids))
        self.total_batch_num = int(len(self.Entity_var_ids) / self.batch_size)



    def load_data(self, entity_file, context_file, feature_file, entity_type_feature_file, entity_type_exact_feature_file, type_file, window_size = 10):
        with open('all_types.txt', 'r') as f:
            self.all_types = [data.replace('\n','') for data in f]
        with open('held_out_types.txt', 'r') as f:
            self.held_out_types = [data.replace('\n','') for data in f]


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

        self.Features = np.load(feature_file)

        self.Entity_type_features = np.load(entity_type_feature_file)

        # for i in range(0, self.Entity_type_features.shape[0]):
        #     for j in range(0, self.Entity_type_features.shape[1]):
        #         for k in range(0, 3):
        #             if self.Exact_entity_type_features[i][j][k] > 0:
        #                 self.Exact_entity_type_features[i][j][k] = 1


        self.Exact_entity_type_features = np.load(entity_type_exact_feature_file)

        self.Type_only_features = np.load('./data/webisa_type_only_features.npy')

        for i in range(0, self.Type_only_features.shape[0]):
            self.Type_only_features[i][0] = np.log(self.Type_only_features[i][0]+1.0)

        self.r = list(range(0, len(self.Entity_var_ids)))

        if self.shuffle_flag == 1:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.r)


    def gen_random_pos_emb(self, pos_emb_len = 50):
        self.static_pos_emb = {}
        for i in range(-100, 100):
            self.static_pos_emb[i] = np.random.random(pos_emb_len)


    # select_flag == 0: select_all
    # select_flag == 1: select train label, delete ids
    # select_flag == 2: select test label, take ids

    # def next_batch(self, window_size = 10, select_flag = 0, seen_label_ids = None, unseen_label_ids = None):
    def next_batch(self, select_label_ids):
        ret_Entity_var_ids = []
        ret_r_Left_context_ids = []
        ret_Right_context_ids = []
        ret_Feature_ids = []
        ret_Entity_type_features = []
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
            local_Features = self.Features[self.r[self.train_pos]]
            local_Entity_type_features = self.Entity_type_features[self.r[self.train_pos]]
            local_Exact_entity_type_features = self.Exact_entity_type_features[self.r[self.train_pos]]
            # for i in range(0, len(local_Exact_entity_type_features)):
            #     for j in range(0,3):
            #         if local_Exact_entity_type_features[i][j] > 0:
            #             local_Exact_entity_type_features[i][j] = 1.0
            local_Ys = self.Types[self.r[self.train_pos]]

            ret_Entity_var_ids.append(local_Entity_var_ids)
            ret_r_Left_context_ids.append(local_r_Left_context_ids)
            ret_Right_context_ids.append(local_Right_context_ids)
            ret_Feature_ids.append(local_Features)
            ret_Entity_type_features.append(local_Entity_type_features)
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

        ret_Left_context_ids = np.asarray(ret_Left_context_ids, dtype=np.float32)
        ret_Left_context_lens = np.asarray(ret_Left_context_lens, dtype=np.float32)

        ret_Right_context_ids = np.asarray(ret_Right_context_ids, dtype=np.float32)
        ret_Right_context_lens = np.asarray(ret_Right_context_lens, dtype=np.float32)

        ret_Feature_ids = np.asarray(ret_Feature_ids, dtype=np.int32)
        t_type_only_features = np.tile(self.Type_only_features, [ret_Entity_ids.shape[0], 1, 1])

        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)


        ret_Entity_type_features = np.take(ret_Entity_type_features, select_label_ids, 1)
        ret_Exact_entity_type_features = np.take(ret_Exact_entity_type_features, select_label_ids, 1)
        ret_Type_only_features = np.take(t_type_only_features, select_label_ids, 1)
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
    a = figer_data_multi_label()
    ret = a.next_batch(list(range(0, 113)))
    print('entity')
    print(' '.join(a.vob.i2w(int(e)) for e in ret[0][2]))
    # print(ret[-1][0])
    print('left context')
    print(' '.join(a.vob.i2w(int(e)) for e in ret[2][2]))
    print('right context')
    print(' '.join(a.vob.i2w(int(e)) for e in ret[4][2]))
    print('type')
    dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    for i in range(0, 113):
        if ret[-1][2][i] == 1:
            print(dicts['id2label'][i])



if __name__ == "__main__":
    get_an_example()
