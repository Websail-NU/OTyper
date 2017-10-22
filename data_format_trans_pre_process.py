from sklearn.externals import joblib
import sys


class Batcher:
    def __init__(self,storage,data, dicts):
        self.storage = storage
        self.data = data
        self.dicts = dicts
        self.reduce_sum = 0


    def create_input_output(self,row):
        s_start = row[0]
        s_end = row[1]
        e_start = row[2]
        e_end = row[3]
        labels = row[74:]
        features = row[4:74]
        # seq_context = np.zeros((self.context_length*2 + 1,self.dim))
        # temp = [ self.id2vec[self.storage[i]][:self.dim] for i in range(e_start,e_end)]

        entity = ' '.join([self.dicts['id2word'][self.storage[i]] for i in range(e_start,e_end)])
        context = ' '.join([self.dicts['id2word'][self.storage[i]] for i in range(s_start, s_end)])


        labels_str = ' '.join([self.dicts['id2label'][i] for i in range(0, 113) if labels[i]==1])

        return entity, context, e_start  - s_start, e_end  - s_start, features, labels_str


    def trans_all(self):
        with open('state_of_the_art_dev_data.txt', 'w') as f:
            for row in self.data:
                entity, context, e_start, e_end, features, labels = self.create_input_output(row)
                f.write('{}\n'.format(entity))
                f.write('{}\n'.format(context))
                f.write('{}\n'.format(e_start))
                f.write('{}\n'.format(e_end))
                feature_array = []
                for e in features:
                    feature_array.append(str(e))
                feature_str = ' '.join(feature_array)
                f.write('{}\n'.format(feature_str))
                f.write('{}\n'.format(labels))



def trans_state_of_the_art_to_my_format():
    dicts = joblib.load("/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl")

    test_dataset = joblib.load("/home/zys133/knowledge_base/NFGEC/data/Wiki/dev_figer.pkl")
    print test_dataset['data'].shape[0]
    a = Batcher(test_dataset['storage'], test_dataset['data'], dicts)
    a.trans_all()

if __name__ == "__main__":
    trans_state_of_the_art_to_my_format()
