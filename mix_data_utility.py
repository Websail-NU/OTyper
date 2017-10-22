import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz


def context_and_word_combine():
    with open('mix_data/mix_tagged_context.txt', 'w') as wf:
        with open('data/state_of_the_art_train_tagged_context.txt') as f:
            for line in f:
                wf.write(line)
    with open('mix_data/mix_tagged_context.txt', 'a') as wf:
        with open('umls_data/refined_umls_tagged_context.txt') as f:
            for line in f:
                wf.write(line)
    with open('mix_data/mix_tagged_context.txt', 'a') as wf:
        with open('neural_science_data/nn_tagged_context.txt') as f:
            for line in f:
                wf.write(line)


    with open('mix_data/mix_word_with_context.txt', 'w') as wf:
        with open('data/state_of_the_art_train_word_with_context.txt') as f:
            for line in f:
                wf.write(line)
    with open('mix_data/mix_word_with_context.txt', 'a') as wf:
        with open('umls_data/refined_umls_word.txt') as f:
            for line in f:
                wf.write(line)
    with open('mix_data/mix_word_with_context.txt', 'a') as wf:
        with open('neural_science_data/nn_words.txt') as f:
            for line in f:
                wf.write(line)


def word_list_combine():
    total_word_set = set()
    with open('data/word_list.txt') as f:
        for line in f:
            line = line.strip()
            total_word_set.add(line)
    with open('umls_data/word_list.txt') as f:
        for line in f:
            line = line.strip()
            total_word_set.add(line)
    with open('neural_science_data/word_list.txt') as f:
        for line in f:
            line = line.strip()
            total_word_set.add(line)

    with open('mix_data/word_list.txt', 'w') as wf:
        for e in total_word_set:
            wf.write('{}\n'.format(e))


def type_emb_combine():
    # emb combine
    total_type_emb = None

    figer_type_emb = np.load('data/labelid2emb.npy')
    total_type_emb = figer_type_emb

    umls_type_emb = np.load('umls_data/umls_labelid2emb.npy')
    total_type_emb = np.vstack((total_type_emb, umls_type_emb))

    nn_type_emb = np.load('neural_science_data/nn_labelid2emb.npy')
    total_type_emb = np.vstack((total_type_emb, nn_type_emb))

    print('{} {} {}'.format(len(figer_type_emb), len(umls_type_emb), len(nn_type_emb)))
    print(total_type_emb.shape)
    print(len(figer_type_emb)+len(umls_type_emb))



    # np.save('mix_data/mix_labelid2emb', total_type_emb)


def type_index_combine():
    figer_type_emb = np.load('data/labelid2emb.npy')
    umls_type_emb = np.load('umls_data/umls_labelid2emb.npy')
    nn_type_emb = np.load('neural_science_data/nn_labelid2emb.npy')

    figer_range = range(0, len(figer_type_emb))
    umls_range = range(len(figer_type_emb), len(figer_type_emb) + len(umls_type_emb))
    nn_range = range(len(figer_type_emb) + len(umls_type_emb), len(figer_type_emb) + len(umls_type_emb) + len(nn_type_emb))

    figer_type = np.load('data/state_of_the_art_train_Types_with_context.npy')
    umls_type = np.load('umls_data/umls_Types_with_context.npy')
    nn_type = np.load('neural_science_data/nn_Types_with_context.npy')

    mix_Type = np.zeros((len(figer_type) + len(umls_type) + len(nn_type), len(figer_type_emb) + len(umls_type_emb) + len(nn_type_emb)))

    print(mix_Type.shape)

    p = 0
    for i in range(0, len(figer_type)):
        mix_Type[p][0:len(figer_type_emb)] = figer_type[i]
        p += 1

    for i in range(0, len(umls_type)):
        mix_Type[p][len(figer_type_emb):len(figer_type_emb)+len(umls_type_emb)] = umls_type[i]
        p += 1

    for i in range(0, len(nn_type)):
        mix_Type[p][len(figer_type_emb)+len(umls_type_emb):len(figer_type_emb)+len(umls_type_emb)+len(nn_type_emb)] = nn_type[i]
        p += 1

    np.save('mix_data/mix_Types_with_context', mix_Type)



    # get total number of types and number of type break down

    # for FIGER add 0000 in the end
    # for MSH add 000 = 113 in the beginning and end
    # for NN add 000 in front



# word list combine, done
# word embedding gen, done
# NN type embedding gen, done
# type embedding concatenate, done
# index concatenate, done


def split_train_test():
    figer_type = np.load('data/state_of_the_art_train_Types_with_context.npy')
    umls_type = np.load('umls_data/umls_Types_with_context.npy')
    nn_type = np.load('neural_science_data/nn_Types_with_context.npy')

    train_len = len(figer_type) + len(umls_type)

    test_len = len(nn_type)

    print('splitting word file')

    count = 0
    with open('mix_data/train_mix_word_with_context.txt', 'w') as wf:
        with open('mix_data/mix_word_with_context.txt') as f:
            for line in f:
                if count < train_len:
                    wf.write(line)
                else:
                    break
                count += 1

    count = 0
    with open('mix_data/test_mix_word_with_context.txt', 'w') as wf:
        with open('mix_data/mix_word_with_context.txt') as f:
            for line in f:
                if count < train_len:
                    count += 1
                    continue
                else:
                    wf.write(line)

    print('splitting context file')


    count = 0
    with open('mix_data/train_mix_tagged_context.txt', 'w') as wf:
        with open('mix_data/mix_tagged_context.txt') as f:
            for line in f:
                if count < train_len:
                    wf.write(line)
                else:
                    break
                count += 1

    count = 0
    with open('mix_data/test_mix_tagged_context.txt', 'w') as wf:
        with open('mix_data/mix_tagged_context.txt') as f:
            for line in f:
                if count < train_len:
                    count += 1
                    continue
                else:
                    wf.write(line)

    print('loading numpy file')

    mix_Types = np.load('mix_data/mix_Types_with_context.npy')

    print('start splitting np')

    train_mix_Types = mix_Types[0:train_len]

    test_mix_Types = mix_Types[train_len:]

    print(train_mix_Types.shape)
    print(test_mix_Types.shape)

    np.save('mix_data/train_mix_Types_with_context', train_mix_Types)
    np.save('mix_data/test_mix_Types_with_context', test_mix_Types)


def trans_type_to_sparse_matrix():
    # mix_Type = np.load('mix_data/mix_Types_with_context.npy')
    # sparse_mix_Type = csr_matrix(mix_Type)
    # np.save('mix_data/mix_Types_with_context_sparse', sparse_mix_Type)

    train_mix_Types = np.load('mix_data/train_mix_Types_with_context.npy')
    sparse_train_mix_Types = csr_matrix(train_mix_Types)
    save_npz('mix_data/train_mix_Types_with_context_sparse', sparse_train_mix_Types)

    test_mix_Types = np.load('mix_data/test_mix_Types_with_context.npy')
    sparse_test_mix_Types = csr_matrix(test_mix_Types)
    save_npz('mix_data/test_mix_Types_with_context_sparse', sparse_test_mix_Types)

    # Types = np.load('mix_data/test_mix_Types_with_context_sparse.npy')
    #
    # m = np.zeros((1000,11))
    # sparse_m = csr_matrix(m)
    # save_npz('1000_sparse', sparse_m)
    # sparse_m = load_npz('1000_sparse.npz')
    # print(sparse_m.shape)









def temp():
    pass


if __name__ == "__main__":
    # context_and_word_combine()
    # split_train_test()
    trans_type_to_sparse_matrix()
