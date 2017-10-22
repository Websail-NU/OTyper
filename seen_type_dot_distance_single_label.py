import tensorflow as tf
import numpy as np
import figer_data_single_label_batcher
from operator import itemgetter
import evaluate
from sklearn.externals import joblib
import cPickle as pickle

# label id place holder, done
# label id 2 emb untrainable lookup table, done
# label id 2 emb lookup, done
# dot product, done
# loss function, done
# batcher return value add, done
# place holder feed_dict, done


def seen_type_dot_distance_single_label():
    unseen_label_ids = [91, 12, 26, 63, 36, 79, 51, 90, 88, 72]


    # unseen_label_ids = [3]
    seen_label_ids = []
    for i in range(0,113):
        if not i in unseen_label_ids:
            seen_label_ids.append(i)

    window_size = 10
    entity_ids = tf.placeholder(tf.int32, [None, None], name = 'entity_ids')
    entity_raw_lens = tf.placeholder(tf.float32, [None], name = 'entity_raw_lens')
    l_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'l_context_ids')
    l_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'l_context_raw_lens')
    r_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'r_context_ids')
    r_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'r_context_raw_lens')
    label_ids = tf.placeholder(tf.int32, [None], name = 'label_ids')

    word_emb = np.load('./data/word_emb.npy')
    word_emb_lookup_table = tf.Variable(word_emb, dtype=tf.float32, trainable = False, name = 'word_emb_lookup_table')

    with open('data/labelid2emb.pkl', 'r') as f:
        label_id2emb = pickle.load(f)

    label_id2emb_lookup_table = tf.Variable(label_id2emb, dtype=tf.float32, trainable = False, name = 'label_id2emb_lookup_table')

    l_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, l_context_ids)
    r_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, r_context_ids)
    entity_embs = tf.nn.embedding_lookup(word_emb_lookup_table, entity_ids)

    label_ave_embs = tf.nn.embedding_lookup(label_id2emb_lookup_table, label_ids)

    entity_lens = tf.reshape(entity_raw_lens, [-1, 1], name = 'entity_lens')
    l_context_lens = tf.reshape(l_context_raw_lens, [-1, 1], name = 'l_context_lens')
    r_context_lens = tf.reshape(r_context_raw_lens, [-1, 1], name = 'r_context_lens')

    drop_out = tf.placeholder(tf.float32)

    l_context_embs_sum = tf.reduce_sum(l_context_embs, 1)
    r_context_embs_sum = tf.reduce_sum(r_context_embs, 1)
    context_embs_sum = tf.concat([l_context_embs_sum, r_context_embs_sum], 1, name = 'context_embs_ave')

    entity_embs_sum = tf.reduce_sum(entity_embs, 1)
    entity_embs_ave = entity_embs_sum / entity_lens

    entity_embs_ave_dropout = tf.nn.dropout(entity_embs_ave, 1.0 - drop_out)

    Xs = tf.concat([entity_embs_ave_dropout,context_embs_sum], 1)
    Ys = tf.placeholder(tf.float32, [None, 1], name = 'Ys')

    W = tf.Variable(tf.random_uniform([900, 300], minval=-0.01, maxval=0.01), name = 'W')
    input_representation = tf.matmul(Xs, W)

    logit = tf.reduce_sum(input_representation * label_ave_embs, 1, keep_dims=True)

    predict_y = tf.sigmoid(logit)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Ys, logits = logit))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    figer = figer_data_single_label_batcher.figer_data_single_label(entity_file = 'data/state_of_the_art_train_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_train_tagged_context.txt', \
                    type_file = 'data/state_of_the_art_train_Types_with_context.npy', negative_sampling_rate = 1.0)
    figer_test = figer_data_single_label_batcher.figer_data_single_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy', negative_sampling_rate = 1.0)

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, 5):
            figer.shuffle()
            np_store = []
            for i in range(0, 2000):
                batch_entity_ids, batch_entity_raw_lens, batch_l_context_ids, batch_l_context_lens, \
                                batch_r_context_ids, batch_r_context_lens, batch_label_ids, batch_ys \
                                    = figer.next_batch(select_lable_ids=seen_label_ids)
                _, print_loss = \
                sess.run([train_step, loss],\
                 feed_dict={entity_ids: batch_entity_ids, entity_raw_lens: batch_entity_raw_lens, \
                            l_context_ids: batch_l_context_ids, l_context_raw_lens: batch_l_context_lens, \
                            r_context_ids: batch_r_context_ids, r_context_raw_lens: batch_r_context_lens, \
                            label_ids: batch_label_ids, Ys: batch_ys, drop_out: [0.5]})

                with open('temp.txt', 'w') as f:
                    f.write('training epoch: {} batch {} : {}\n'.format(epoch, i, print_loss))

            predict_ys = np.zeros([0, len(seen_label_ids)])
            truth_ys = np.zeros([0, len(seen_label_ids)])

            for i in range(0, 200):
                batch_entity_ids, batch_entity_raw_lens, batch_l_context_ids, batch_l_context_lens, \
                                batch_r_context_ids, batch_r_context_lens, batch_label_ids, batch_ys \
                                    = figer.next_batch(select_lable_ids=seen_label_ids)
                print_predict_y = \
                sess.run(predict_y,\
                 feed_dict={entity_ids: batch_entity_ids, entity_raw_lens: batch_entity_raw_lens, \
                            l_context_ids: batch_l_context_ids, l_context_raw_lens: batch_l_context_lens, \
                            r_context_ids: batch_r_context_ids, r_context_raw_lens: batch_r_context_lens, \
                            label_ids: batch_label_ids, Ys: batch_ys, drop_out: [0.0]})
                print_predict_y = print_predict_y.reshape((-1, len(seen_label_ids)))
                batch_ys = batch_ys.reshape((-1, len(seen_label_ids)))
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_ys))
            evaluate.acc_hook(predict_ys, truth_ys, epoch, 0)

            figer_test.train_pos = 0
            batch_entity_ids, batch_entity_raw_lens, batch_l_context_ids, batch_l_context_lens, \
                            batch_r_context_ids, batch_r_context_lens, batch_label_ids, batch_ys \
                             = figer_test.next_batch(select_lable_ids=unseen_label_ids)
            print_entity_embs_ave, print_predict_y = sess.run([entity_embs_ave, predict_y],\
             feed_dict={entity_ids: batch_entity_ids, entity_raw_lens: batch_entity_raw_lens, \
                        l_context_ids: batch_l_context_ids, l_context_raw_lens: batch_l_context_lens, \
                        r_context_ids: batch_r_context_ids, r_context_raw_lens: batch_r_context_lens, \
                        label_ids: batch_label_ids, drop_out: [0.0]})

            print_predict_y = print_predict_y.reshape((-1, len(unseen_label_ids)))
            batch_ys = batch_ys.reshape((-1, len(unseen_label_ids)))

            evaluate.acc_hook(print_predict_y, batch_ys, epoch, 0)


def temp():
    figer = figer_data_single_label_batcher.figer_data_single_label(entity_file = 'data/state_of_the_art_train_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_train_tagged_context.txt', \
                    type_file = 'data/state_of_the_art_train_Types_with_context.npy')
    batch_entity_ids, batch_entity_lens, batch_l_context_ids, batch_l_context_lens, \
    batch_r_context_ids, batch_r_context_lens, batch_label_ids, batch_ys = figer.next_batch()

    print batch_entity_ids.shape
    print batch_entity_lens.shape
    print batch_l_context_ids.shape
    print batch_l_context_lens.shape
    print batch_r_context_ids.shape
    print batch_r_context_lens.shape
    print batch_label_ids.shape
    print batch_ys.shape


if __name__ == "__main__":
    seen_type_dot_distance_single_label()
