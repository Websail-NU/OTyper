import json
import cPickle as pickle
import gensim
import arff
import string
import os
import numpy as np
import sys
import umls_data_batcher



def get_all_cui():
    ret = []
    with open('/websail/common/umls/MSHCorpus/benchmark_mesh.txt', 'r') as f:
        for line in f:
            for e in line.split():
                if e[0] == 'C' and e[1].isdigit():
                    ret.append(e)
    return ret


def cui_to_gid(cui, entities):
    for e in entities:
        if cui in e['research_entity_id']:
            return e['global_id']

    return -1


def gid_to_index(gid):
    l = get_all_gid()
    return l.index(gid)


def expand_gid(gid, all_entity_gid_list):
    for l in all_entity_gid_list:
        if l[0] == gid:
            return l

    assert(False == True)

    return None


def gen_all_gid():
    with open('umls_data/all_entity_gid_list.pkl', 'r') as f:
        all_entity_gid_list = pickle.load(f)
    with open('umls_data/all_entity_name_list.pkl', 'r') as f:
        all_entity_name_list = pickle.load(f)

    d = {}
    assert(len(all_entity_gid_list) == len(all_entity_name_list))
    for i in range(0, len(all_entity_gid_list)):
        assert(len(all_entity_gid_list[i]) == len(all_entity_name_list[i]))
        for j in range(0, len(all_entity_gid_list[i])):
            d[all_entity_gid_list[i][j]] = all_entity_name_list[i][j]


    all_entity_gid_list_new = []
    all_entity_name_list_new = []

    for k in d:
        all_entity_gid_list_new.append(k)
        all_entity_name_list_new.append(d[k])

    print len(all_entity_gid_list_new)

    with open('umls_data/all_entity_gid_list_new.pkl', 'w') as wf:
        pickle.dump(all_entity_gid_list_new, wf)

    with open('umls_data/all_entity_name_list_new.pkl', 'w') as wf:
        pickle.dump(all_entity_name_list_new, wf)


def get_all_gid():
    with open('umls_data/all_entity_gid_list_new.pkl', 'r') as f:
        all_entity_gid_list = pickle.load(f)
    return all_entity_gid_list



def get_full_set_cui():
    cui_list = get_all_cui()
    with open('relations.pkl', 'r') as f:
        relations = pickle.load(f)

    for e in relations:
        if 'type' in e['relation_type']:
            print e
            break



def filter_umls_file_entity():
    cuis = get_all_cui()
    entities = []
    with open('/websail/common/umls/umls.json', 'r') as f:
        umls_data = json.load(f)

    count = 0
    flag = 0
    for e in umls_data['entities']:
        for c in cuis:
            if c in e['research_entity_id']:
                entities.append(e)
                if flag == 0:
                    print e['research_entity_id']
                    print c
                    flag = 1
                break
        count += 1
        if count % 10000 == 0:
            print count

    with open('umls_data/entities.pkl', 'w') as f:
        pickle.dump(entities,f)
    print len(entities)


def filter_umls_file_relation():
    with open('entities.pkl', 'r') as f:
        entities = pickle.load(f)

    r_set = set()

    relations = []
    for e in entities:
        r_set.update(e['relation_ids'])

    with open('/websail/common/umls/umls.json', 'r') as f:
        umls_data = json.load(f)

    print 'total = {}'.format(len(umls_data['relations']))
    print len(r_set)
    count = 0
    for e in umls_data['relations']:
        if e['global_id'] in r_set:
            relations.append(e)
        count += 1
        if count % 10000 == 0:
            print count

    print len(relations)

    with open('relations.pkl', 'w') as f:
        pickle.dump(relations,f)


# gen all parent entities and gid list

def filter_umls_file_relation_new():
    with open('umls_data/entities.pkl', 'r') as f:
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
        print count
        count += 1

        all_entity_gid_list.append(entity_gid_list)
        all_entity_name_list.append(entity_name_list)


    with open('umls_data/all_entity_gid_list.pkl', 'w') as f:
        pickle.dump(all_entity_gid_list,f)
    with open('umls_data/all_entity_name_list.pkl', 'w') as f:
        pickle.dump(all_entity_name_list,f)



def get_parent_gid(c_gid, relations, r_dict):
    if c_gid in r_dict:
        return relations[r_dict[c_gid]]['entity_ids'][0]

    return -1


def gen_umls_data():
    reload(sys)
    sys.setdefaultencoding('utf8')

    black_list = ['C0032305', 'C0927232', 'C0043117']

    Types = []
    with open('umls_data/all_entity_gid_list.pkl', 'r') as f:
        all_entity_gid_list = pickle.load(f)
    with open('umls_data/entities.pkl', 'r') as f:
        entities = pickle.load(f)
    with open('umls_data/umls_tagged_context.txt', 'w') as wf:
        for file_name in os.listdir('/websail/common/umls/MSHCorpus'):
            if file_name == 'README.txt' or file_name == 'benchmark_mesh.txt':
                continue
            with open(os.path.join('/websail/common/umls/MSHCorpus', file_name), 'r') as f:
                id_l = f.next().split()[1].split('_')
                if len(set(id_l).intersection(black_list)):
                    continue
                for i in range(0, 6):
                    f.next()
                for line in f:
                    context, ids = get_context_and_class(line, id_l, entities, all_entity_gid_list)
                    if ids[0] == -1:
                        continue
                    wf.write('{}\n'.format(' '.join(context)))
                    local_Type = np.zeros(1387)
                    for index in ids:
                        local_Type[index] = 1.0
                    Types.append(local_Type)

    np.save('umls_data/umls_Types_with_context', Types)


# gen entity tagged context file and word list file
def w2v_gen_umls_tagged_contex():
    w2v = gensim.models.KeyedVectors.load_word2vec_format('/websail/common/embeddings/glove/840B/glove.840B.300d.bin', binary=True)
    word_set = set()
    with open('umls_data/refined_umls_tagged_context.txt', 'w') as wf:
        with open('umls_data/umls_tagged_context.txt', 'r') as f:
            for line in f:
                wa = []
                for e in line.split(' '):
                    if e in w2v:
                        wa.append(e)
                        word_set.add(e)
                    elif e.lower() in w2v:
                        wa.append(e.lower())
                        word_set.add(e.lower())
                    else:
                        wa.append(e)
                        word_set.add('unk')
                wf.write(' '.join(wa))

    word_set.add('_my_null_')

    with open('umls_data/entities.pkl', 'r') as f:
        entities = pickle.load(f)
    with open('umls_data/all_entity_name_list.pkl', 'r') as f:
        all_entity = pickle.load(f)
    for l in all_entity:
        for entity in l:
            for e in entity.split(' '):
                if e in w2v:
                    word_set.add(e)
                elif e.lower() in w2v:
                    word_set.add(e.lower())

    with open('umls_data/word_list.txt', 'w') as wf:
        for e in list(word_set):
            wf.write('{}\n'.format(e))


def gen_type_emb():
    word_emb = np.load('./umls_data/umls_word_emb.npy').astype(np.float32)

    with open('umls_data/all_entity_name_list_new.pkl', 'r') as f:
        all_entity_name_list_flat = pickle.load(f)

    vob = umls_data_batcher.Vocabulary()

    ave_type_emb_list = []
    for t in all_entity_name_list_flat:
        emb_list = []
        for w in t.split(' '):
            word_index = vob.w2i(w)
            emb_list.append(word_emb[word_index])
        if len(emb_list) == 0:
            print 'zero length'
            ave_type_emb_list.append(np.zeros(300))
        else:
            ave_type_emb_list.append(np.mean(emb_list, axis=0))

    ave_type_emb_list = np.asarray(ave_type_emb_list, dtype=np.float32)
    print ave_type_emb_list.shape

    np.save('umls_data/umls_labelid2emb', ave_type_emb_list)

    # with open('umls_data/umls_labelid2emb.pkl', 'w') as wf:
    #     pickle.dump(ave_type_emb_list, wf)


def gen_entity_word_file():
    count = 0
    with open('umls_data/refined_umls_word.txt', 'w') as wf:
        with open('umls_data/refined_umls_tagged_context.txt', 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                words = line.split(' ')
                start_i = words.index('<e>')
                count += 1
                end_i = words.index('</e>')
                wf.write('{}\n'.format(' '.join(words[(start_i+1):end_i])))


def get_context_and_class(line, s_cui_list, entities, all_entity_gid_list):
    assert(line[-4]==',')
    id = int(line[-2]) - 1
    context_start_index = line.index(',') + 2
    tok_context = my_tokenier(line[context_start_index:-5], s_cui_list[id], entities)
    gid = cui_to_gid(s_cui_list[id], entities)
    if gid == -1:
        return None, [-1]

    gids = expand_gid(gid, all_entity_gid_list)

    indices = []
    for gid in gids:
         indices.append(gid_to_index(gid))

    return tok_context, indices


def my_tokenier(line, id, entities):
    ret = []
    in_flag = 0
    for e in line.split():
        if in_flag == 1:
            if '</e>' in e:
                in_flag = 0
                ret.append('</e>')
        elif '<e>' in e and in_flag == 0:
            ret.append('<e>')
            ret += trans_id_to_entities(id, entities)
            if not '</e>' in e:
                in_flag = 1
            else:
                ret.append('</e>')
        else:
            ret.append(unicode(e.translate(None, string.punctuation), errors='ignore'))

    return ret



def trans_id_to_entities(id, entities):
    for e in entities:
        if id in e['research_entity_id']:
            name = e['canonical_name']
            return name.split()

    print id
    assert(True == False)
    return None



def temp():
    with open('/websail/common/umls/umls.json', 'r') as f:
        umls_data = json.load(f)
    for e in umls_data['entities']:
        if e['canonical_name'] == 'age' or e['canonical_name'] == 'few':
            print e


if __name__ == "__main__":
    # filter_umls_file_relation()
    # gen_umls_data()
    # w2v_gen_umls_tagged_contex()
    # get_full_set_cui()
    # filter_umls_file_relation_new()
    temp()
    # gen_all_gid()
    # gen_entity_word_file()
    # gen_type_emb()
