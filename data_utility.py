import cPickle as pickle


#   input: the txt figer data
#   output: Xs and Ys for training, X is [w2v, type_w2v], Y = [0] or [1]

def read_Mentions_embedding_types():
    all_types = ['actor', 'doctor']
    held_out_types = ['doctor', 'government', 'county', 'camera', 'music', 'airport', 'terrorist_attack', 'drug', 'algorithm', 'law']
    with open('/websail/common/figer_data/figer_trans_to_python.txt', 'r') as f:
        count = 0
        for line in f:
            if count % 5 ==0:
                raw_entity_string = line.replace('\n', '')
            if count % 5 == 2:
                start = int(line)
            if count % 5 == 3:
                end = int(line)
                entity_embedding = get_entity_embedding(raw_entity_string, start, end, w2v)
            if count % 5 ==4:
                raw_labels_string = line.replace('\n', '')
                matched_labels = get_labels(raw_labels_string)
                if entity_embedding != None:
                    for label in all_types:
                        if label in held_out_types:
                            continue
                        if label in matched_labels:
                            X = entity_embedding, label_embedding
                            Y = 1
                        else:
                            X = entity_embedding, label_embedding
                            Y = 0

            count += 1
            if count > 19:
                break

# input: entity string
# output: entity_embedding

def get_entity_embedding(raw_entity_string, start, end, w2v):
    tokens = raw_entity_string.split(' ')
    ave_emb = np.zeros(300)
    for i in range(0, (end-start)):
        if tokens.lower() in w2v:
            ave_emb += w2v[tokens.lower()]
        else:
            return None

    return ave_emb / float(end-start)

#   input: raw_labels_string
#   output: short_labels

def get_labels(raw_labels_string):
    full_labels = raw_labels_string.split(' ')
    short_labels = [full_label.split('/')[-1] for full_label in full_labels]

    return short_labels

    for l in short_labels:
        if l in held_out_types:
            continue
        tokens = l.split('_')
        ave_emb = np.zeros(300)
        for w in tokens:
            if w.lower() in w2v:
                ave_emb += w2v[w.lower()] / float(len(tokens))
            else:
                ave_emb = None
                break
        if ave_emb != None:
            ret.append(ave_emb)

    return ret

def temp():
    with open('/websail/common/figer_data/figer_trans_to_python.txt', 'r') as f:
        type_set = set()
        count = 0
        for line in f:
            if count % 5 ==4:
                raw_labels_string = line.replace('\n', '')
                matched_labels = get_labels(raw_labels_string)
                # if 'shce021709' in matched_labels:
                #     print raw_labels_string
                #     while(True):
                #         pass
                type_set.update(matched_labels)
                # if len(type_set) > 100000:
                #     print type_set
                #     break
            count += 1

        print len(type_set)

if __name__ == "__main__":
    temp()
