import cPickle as pickle
import numpy as np
import gensim
import random


class figer_data:
    def __init__ (self):
        self.load_data()
        self.train_shuffle_pos = 0
        self.train_order_pos = 0
        self.test_order_pos = 0
        self.batch_size = 1000
        self.total_batch_num = int(len(self.Words) / self.batch_size)

    def load_data(self):
        with open('all_types.txt', 'r') as f:
            self.all_types = [data.replace('\n','') for data in f]
        with open('held_out_types.txt', 'r') as f:
            self.held_out_types = [data.replace('\n','') for data in f]

        self.Words = []
        with open('data/word.txt', 'r') as f:
            for line in f:
                self.Words.append(line.replace('\n',''))

        self.Contexts = []
        with open('data/context.txt', 'r') as f:
            for line in f:
                self.Contexts.append(line.replace('\n',''))

        self.Types = np.load('data/Types.npy')
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

        self.r = range(0, len(self.Words)*113)

        random.shuffle(self.r)

        # for i in range(0, len(self.Words)):
        #     t = self.Words[i]
        #     self.Words[i] = self.Words[r[i]]
        #     self.Words[r[i]] = t
        #     t = self.Contexts[i]
        #     self.Contexts[i] = self.Contexts[r[i]]
        #     self.Contexts[r[i]] = t
        #     t = self.Types[i]
        #     self.Types[i] = self.Types[r[i]]
        #     self.Types[r[i]] = t

    def next_shuffle_train_batch(self):
        ret_Xs = []
        ret_Ys = []
        ret_Ws = []
        #
        # for i in range(0, 100):
        #     ret_Xs.append(np.zeros(600))
        #     ret_Ys.append([1])
        # ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        # ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        #
        # return ret_Xs, ret_Ys

        count = 0
        while count<=self.batch_size*113:
            word_index = self.r[self.train_shuffle_pos] / 113
            type_index = self.r[self.train_shuffle_pos] % 113
            if self.all_types[type_index] in self.held_out_types:
                self.train_shuffle_pos = (self.train_shuffle_pos + 1) % len(self.Words)*113
                continue
            entity_embedding = get_entity_embedding(self.Words[word_index], self.w2v)
            label_embedding = get_label_embedding(self.all_types[type_index], self.w2v)
            local_X = np.concatenate((entity_embedding, label_embedding))
            if self.Types[word_index][type_index] == 1:
                local_Y = 1
                local_W = 0.973278765
            else:
                local_Y = 0
                local_W = 1.0 - 0.973278765
            ret_Xs.append(local_X)
            ret_Ys.append([local_Y])
            ret_Ws.append([local_W])
            self.train_shuffle_pos = (self.train_shuffle_pos + 1) % len(self.Words)*113
            count += 1

        ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        ret_Ws = np.asarray(ret_Ws, dtype=np.float32)

        return ret_Xs, ret_Ys, ret_Ws

    def next_order_train_batch(self):
        ret_Xs = []
        ret_Ys = []
        ret_Ws = []
        #
        # for i in range(0, 100):
        #     ret_Xs.append(np.zeros(600))
        #     ret_Ys.append([1])
        # ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        # ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        #
        # return ret_Xs, ret_Ys

        count = 0
        while count<=self.batch_size:
            entity_embedding = get_entity_embedding(self.Words[self.train_order_pos], self.w2v)
            for i in range(0, 113):
                if self.all_types[i] in self.held_out_types:
                    continue
                label_embedding = get_label_embedding(self.all_types[i], self.w2v)
                local_X = np.concatenate((entity_embedding, label_embedding))
                if self.Types[self.train_order_pos][i] == 1:
                    local_Y = 1
                    local_W = 0.973278765
                else:
                    local_Y = 0
                    local_W = 1.0 - 0.973278765
                ret_Xs.append(local_X)
                ret_Ys.append([local_Y])
                ret_Ws.append([local_W])
            self.train_order_pos = (self.train_order_pos + 1) % len(self.Words)
            count += 1

        ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        ret_Ws = np.asarray(ret_Ws, dtype=np.float32)

        return ret_Xs, ret_Ys, ret_Ws

    def next_order_test_batch(self):
        ret_Xs = []
        ret_Ys = []
        ret_Ws = []
        #
        # for i in range(0, 100):
        #     ret_Xs.append(np.zeros(600))
        #     ret_Ys.append([1])
        # ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        # ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        #
        # return ret_Xs, ret_Ys

        count = 0
        while count<=self.batch_size:
            entity_embedding = get_entity_embedding(self.Words[self.test_order_pos], self.w2v)
            for i in range(0, 113):
                if self.all_types[i] in self.held_out_types:
                    label_embedding = get_label_embedding(self.all_types[i], self.w2v)
                    local_X = np.concatenate((entity_embedding, label_embedding))
                    if self.Types[self.test_order_pos][i] == 1:
                        local_Y = 1
                        local_W = 0.973278765
                    else:
                        local_Y = 0
                        local_W = 1.0 - 0.973278765
                    ret_Xs.append(local_X)
                    ret_Ys.append([local_Y])
                    ret_Ws.append([local_W])
            self.test_order_pos = (self.test_order_pos + 1) % len(self.Words)
            count += 1

        ret_Xs = np.asarray(ret_Xs, dtype=np.float32)
        ret_Ys = np.asarray(ret_Ys, dtype=np.float32)
        ret_Ws = np.asarray(ret_Ws, dtype=np.float32)

        return ret_Xs, ret_Ys, ret_Ws



def get_context_seq(context_string, label_embedding, w2v):
    tokens = context_string.split(' ')

    assert(len(tokens)==14)

    ret = []
    for i in range(0,7):
        if tokens[i] == '_my_null_':
            ret.append(np.zeros(300))
        else:
            ret.append(w2v[tokens[i]])

    ret.append(label_embedding)

    for i in range(0,7):
        if tokens[i+7] == '_my_null_':
            ret.append(np.zeros(300))
        else:
            ret.append(w2v[tokens[i+7]])

    return ret



#   input: the txt figer data
#   output: Words.txt  Type.npy

def read_Mentions_embedding_types():
    # stop_words = []
    # with open('data/stopwords_big', 'r') as f:
    #     for line in f:
    #         line = line.replace('\r','')
    #         line = line.replace('\n', '')
    #         stop_words.append(line.lower())

    with open('all_types.txt', 'r') as f:
        all_types = [data.replace('\n','') for data in f]
    with open('held_out_types.txt', 'r') as f:
        held_out_types = [data.replace('\n','') for data in f]
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

    Words = []
    Types = []
    Contexts = []

    with open('/websail/common/figer_data/figer_trans_to_python.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 100000 == 0:
                print count
            if count % 5 ==0:
                raw_entity_string = line.replace('\n', '')
            if count % 5 == 1:
                context_string = line.replace('\n', '')
            if count % 5 == 2:
                start = int(line)
            if count % 5 == 3:
                end = int(line)
                entity_refined_string = get_entity_string(raw_entity_string, start, end, w2v)
            if count % 5 ==4:
                raw_labels_string = line.replace('\n', '')
                matched_labels = get_labels(raw_labels_string, all_types)
                if not entity_refined_string is None:
                    previous_after_words = get_previous_after_words(context_string, start, end, w2v, 7)
                    local_Types = np.zeros(113)
                    correct_count = 0
                    for i in range(0, 113):
                        label = all_types[i]
                        # if label in held_out_types:
                        #     continue
                        if label in matched_labels:
                            local_Types[i] = 1
                            correct_count += 1

                    if correct_count > 0:
                        Words.append(entity_refined_string)
                        Contexts.append(' '.join(previous_after_words))
                        Types.append(local_Types)

            count += 1

    print len(Words)
    print len(Types)
    print len(Contexts)

    with open('data/word_with_context.txt', 'w') as f:
        for word in Words:
            f.write("{}\n".format(word))
    with open('data/context.txt', 'w') as f:
        for line in Contexts:
            f.write("{}\n".format(line))

    np.save('data/Types_witn_context', Types)


def get_previous_after_words(context_string, start, end, w2v, window_size):
    context_string = context_string.lower()
    tokens = context_string.split(' ')
    previous_words = []
    after_words = []

    pos = start - 1
    count = window_size

    while count > 0:
        if pos >= 0:
            if tokens[pos] in w2v:
                previous_words.append(tokens[pos])
                count -= 1
            pos -= 1
        else:
            previous_words.append('_my_null_')
            count -= 1
    assert(len(previous_words) == window_size)
    previous_words = previous_words[::-1]

    pos = end
    count = window_size
    while count > 0:
        if pos < len(tokens):
            if tokens[pos] in w2v:
                after_words.append(tokens[pos])
                count -= 1
            pos += 1
        else:
            after_words.append('_my_null_')
            count -= 1
    assert(len(after_words) == window_size)

    ret = previous_words + after_words
    return ret

#   input: short label
#   output: average label embedding

def get_label_embedding(label, w2v):
    tokens = label.split('_')
    ave_emb = np.zeros(300)
    for token in tokens:
        if token in w2v:
            ave_emb += w2v[token]/float(len(tokens))

    return ave_emb


# input: refined entity string
# output: entity_embedding

def get_entity_embedding(refined_entity_string, w2v):
    tokens = refined_entity_string.split(' ')

    ave_emb = np.zeros(300)
    for i in range(0, len(tokens)):
        if tokens[i].lower() in w2v:
            ave_emb += w2v[tokens[i].lower()]
        else:
            return None

    return ave_emb / float(len(tokens))

# input: raw_entity string
# output: refined entity string

def get_entity_string(raw_entity_string, start, end, w2v):
    tokens = raw_entity_string.replace(',', '')
    tokens = raw_entity_string.replace('(', '')
    tokens = raw_entity_string.replace(')', '')
    tokens = raw_entity_string.split(' ')

    a = []
    for i in range(0, min(len(tokens),(end-start))):
        if tokens[i].lower() in w2v:
            a.append(tokens[i].lower())
        else:
            return None

    return ' '.join(a)

#   input: raw_labels_string
#   output: short_labels

def get_labels(raw_labels_string, all_types):
    ret = set()

    for raw_e in raw_labels_string.split(' '):
        for e in all_types:
            string_1 = '/' + str(e) + '/'
            if string_1 in raw_e:
                ret.add(e)
            string_2 = '/' + str(e)
            if raw_e[len(raw_e)-len(string_2):] == string_2:
                ret.add(e)
    return ret


def temp():
    stop_words = []
    with open('data/stopwords_big', 'r') as f:
        for line in f:
            line = line.replace('\r','')
            line = line.replace('\n', '')
            stop_words.append(line.lower())

    context_string = 'apple is a table such as cup apple is a table such as cup apple is a table such as cup apple is a table such as cup'

    count = 0
    for token in context_string.split(' '):
        print '{}, {}'.format(token,count)
        count += 1

    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

    ret = get_previous_after_words(context_string, 12, 14, stop_words, w2v, 5)
    print ret

    ret = get_previous_after_words(context_string, 3, 4, stop_words, w2v, 5)
    print ret

    ret = get_previous_after_words(context_string, 26, 27, stop_words, w2v, 5)

    print ret



if __name__ == "__main__":
    a = figer_data()
    X,Y,W = a.get_test_data(0, 10000)

    for i in range(0, X.shape[0]):
        if Y[i] == 1:
            print X[i]
            print Y[i]
            print W[i]

    # read_Mentions_embedding_types()
   # temp()
