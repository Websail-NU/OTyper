import json
import os
import pickle
import numpy as np
import gensim


def gen_entity_sentences():
    entities = get_all_UMLS_entity()
    dir_path = '/websail/large_files/neural_science_corpus_tokenized'
    count = 0
    for file_name in os.listdir(dir_path):
        print(count)
        with open(os.path.join(dir_path, file_name)) as f:
            for line in f:
                for e in entities:
                    indices = find_all(line, e)
                    gen_tagged_sentences(e, line, indices)


def gen_tagged_sentences(entity, line, indices):
    for entity_start in indices:
        if entity_start == -1:
            break
        else:
            content_start = max(entity_start - 200, 0)
            content_end = min(entity_start + len(entity) + 200, len(line))

            content = line[content_start : entity_start+1] + '<e> ' + line[entity_start+1 : entity_start + 1 + len(entity)] \
                + ' </e>' + line[entity_start + 1 + len(entity) : content_end]

            with open('neural_science_data/tagged_sentences.txt', 'a') as f:
                f.write(content+'\n')



def get_all_UMLS_entity():
    if os.path.exists('umls_data/all_entities.json'):
        with open('umls_data/all_entities.json', 'r') as f:
            entities = json.load(f)
        return entities
    else:
        with open('/websail/common/umls/umls.json', 'r') as f:
            umls_data = json.load(f)

        ret_array = []
        for e in umls_data['entities']:
            if len(e['canonical_name'].split()) > 1:
                ret_array.append(e['canonical_name'])
        with open('umls_data/all_entities.json', 'w') as f:
            json.dump(ret_array, f)
        return ret_array


def find_all(a_str, sub):
    start = 0
    ret = []
    a_str = a_str
    sub = ' ' + sub + ' '
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            ret.append(-1)
            break
        else:
            ret.append(start)
            start += len(sub)
    return ret

# gen tagged context file and entity list file

def get_nn_umls_entities():
    entities = []
    with open('neural_science_data/nn_tagged_context.txt', 'w') as wf:
        with open('neural_science_data/tagged_sentences.txt_bk', 'r') as f:
            for line in f:
                line = line.strip()
                words = line.split()
                if(len(line) < 2):
                    continue
                start_index = words.index('<e>')
                end_index = words.index('</e>')
                wf.write('{}\n'.format(line))
                entities.append((' '.join(words[start_index+1 : end_index])))

    with open('neural_science_data/nn_words.txt', 'w') as wf:
        for e in entities:
            wf.write('{}\n'.format(e))

    entities_set = set()

    for e in entities:
        entities_set.add(e)


    with open('/websail/common/umls/umls.json', 'r') as f:
        umls_data = json.load(f)

    entities_array = []
    for e in umls_data['entities']:
        if e['canonical_name'] in entities_set:
            entities_array.append(e)

    with open('neural_science_data/all_nn_entities.pkl', 'wb') as wf:
        pickle.dump(entities_array, wf)


def gen_nn_type_array():
    type_len = get_number_of_types()
    with open('neural_science_data/all_nn_entity_gid_list.pkl', 'rb') as f:
        all_nn_entity_gid_list = pickle.load(f)
    with open('neural_science_data/all_nn_entities.pkl', 'rb') as f:
        nn_entities = pickle.load(f)

    Types = []
    with open('neural_science_data/nn_tagged_context.txt') as f:
        for line in f:
            ids = get_nn_class(line, nn_entities, all_nn_entity_gid_list)
            if ids[0] == -1:
                continue
            local_Type = np.zeros(type_len)
            for index in ids:
                local_Type[index] = 1.0
            Types.append(local_Type)

    np.save('neural_science_data/nn_Types_with_context', Types)


def get_nn_class(line, entities, all_entity_gid_list):
    line = line.strip()
    words = line.split()
    start_index = words.index('<e>')
    end_index = words.index('</e>')
    entity_name = ' '.join(words[start_index+1 : end_index])

    gid = -1
    for e in entities:
        if e['canonical_name'] == entity_name:
            gid = e['global_id']
            break
    if gid == -1:
        print(entity_name)
        for e in entities:
            print(e['canonical_name'])

    gids = expand_gid(gid, all_entity_gid_list)

    indices = []
    for gid in gids:
         indices.append(gid_to_index(gid))

    return indices


def expand_gid(gid, all_entity_gid_list):
    for l in all_entity_gid_list:
        if l[0] == gid:
            return l

    assert(False == True)

    return None


def gid_to_index(gid):
    l = get_all_gid()
    return l.index(gid)


def get_all_gid():
    with open('neural_science_data/all_nn_entity_gid_to_id_list.pkl', 'rb') as f:
        all_entity_gid_list = pickle.load(f)
    return all_entity_gid_list


def gen_nn_gid_to_index():
    with open('neural_science_data/all_nn_entity_gid_list.pkl', 'rb') as f:
        all_entity_gid_list = pickle.load(f)

    s = set()
    for l in all_entity_gid_list:
        for e in l:
            s.add(e)

    w_l = []
    for e in s:
        w_l.append(e)

    with open('neural_science_data/all_nn_entity_gid_to_id_list.pkl', 'wb') as f:
        pickle.dump(w_l, f)




# gen all parent entities and gid list

def filter_nn_file_relation_new():
    with open('neural_science_data/all_nn_entities.pkl', 'rb') as f:
        entities = pickle.load(f)

    r_list = []

    relations = []
    for e in entities:
        r_list.append(e['global_id'])

    with open('/websail/common/umls/umls.json', 'r') as f:
        umls_data = json.load(f)

    entities = umls_data['entities']

    e_dict = {}
    for i, e in enumerate(entities):
        e_dict[e['global_id']] = i
    r_dict = {}
    for i, e in enumerate(umls_data['relations']):
        if e['relation_type'] == 'is_parent_of':
            r_dict[e['entity_ids'][1]] = i

    all_entity_gid_list = []
    all_entity_name_list = []
    count = 0
    for raw_gid in r_list:
        c_gid = raw_gid
        entity_gid_list = []
        entity_name_list = []
        while True:
            entity_gid_list.append(c_gid)
            entity_name_list.append(entities[e_dict[c_gid]]['canonical_name'])
            c_gid = get_parent_gid(c_gid, umls_data['relations'], r_dict)
            if c_gid in [-1]:
                break
            elif entities[e_dict[c_gid]]['canonical_name'] in entity_name_list:
                break
        count += 1

        all_entity_gid_list.append(entity_gid_list)
        all_entity_name_list.append(entity_name_list)


    with open('neural_science_data/all_nn_entity_gid_list.pkl', 'wb') as f:
        pickle.dump(all_entity_gid_list, f)
    with open('neural_science_data/all_nn_entity_name_list.pkl', 'wb') as f:
        pickle.dump(all_entity_name_list, f)


def get_parent_gid(c_gid, relations, r_dict):
    if c_gid in r_dict:
        return relations[r_dict[c_gid]]['entity_ids'][0]

    return -1


def get_number_of_types():
    all_entity_gid_list = get_all_gid()

    return len(all_entity_gid_list)


def gen_nn_word_list():
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)
    word_set = set()
    with open('neural_science_data/nn_tagged_context.txt') as f:
        for line in f:
            line = line.strip()
            for word in line.split(' '):
                if word in w2v:
                    word_set.add(word)

    word_set.add('_my_null_')

    with open('neural_science_data/word_list.txt', 'w') as f:
        for word in word_set:
            f.write('{}\n'.format(word))


def gen_nn_type_emb():
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)
    gid_list = get_all_gid()
    type_emb = []
    for gid in gid_list:
        name = find_name_of_gid(gid)
        local_emb = []
        for e in name.split(' '):
            if e in w2v:
                local_emb.append(w2v[e])
            elif e.lower() in w2v:
                local_emb.append(w2v[e.lower()])
        if len(local_emb) > 0:
            type_emb.append(np.mean(local_emb, axis=0))
        else:
            type_emb.append(np.zeros(300))

    np.save('neural_science_data/nn_labelid2emb', type_emb)
    # with open('neural_science_data/nn_labelid2emb.pkl', 'wb') as f:
    #     pickle.dump(type_emb, f)


def find_name_of_gid(gid):
    with open('neural_science_data/all_nn_entity_gid_list.pkl', 'rb') as f:
        all_entity_gid_list = pickle.load(f)
    with open('neural_science_data/all_nn_entity_name_list.pkl', 'rb') as f:
        all_entity_name_list = pickle.load(f)

    for i in range(0, len(all_entity_gid_list)):
        for j in range(0, len(all_entity_gid_list[i])):
            if all_entity_gid_list[i][j] == gid:
                return all_entity_name_list[i][j]

    assert(False == True)
    return None


def after_tagged_sentence():
    get_nn_umls_entities()
    filter_nn_file_relation_new()
    gen_nn_gid_to_index()
    gen_nn_type_array()
    gen_nn_word_list()
    gen_nn_type_emb()


def get_an_example():
    type_name_list = []
    gid_list = get_all_gid()
    for gid in gid_list:
        name = find_name_of_gid(gid)
        type_name_list.append(name)

    with open('neural_science_data/nn_tagged_context.txt', 'r') as f:
        lines = f.readlines()
    types = np.load('neural_science_data/nn_Types_with_context.npy')

    for example_id in range(0, 100):

        example_type = types[example_id]
        count = 0
        for i in range(0, len(type_name_list)):
            if example_type[i] == 1:
                count += 1
        if count < 2:
            continue

        print('context = {}'.format(lines[example_id]))
        print('type')
        for i in range(0, len(type_name_list)):
            if example_type[i] == 1:
                print(type_name_list[i])





def temp():
    gen_nn_type_emb()



if __name__ == "__main__":
    # after_tagged_sentence()
    # temp()
    get_an_example()
