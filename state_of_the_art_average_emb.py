import tensorflow as tf
import numpy as np
import figer_data_multi_label_batcher
from operator import itemgetter
import evaluate
from sklearn.externals import joblib


def state_of_the_art_average_emb():
    window_size = 10
    entity_ids = tf.placeholder(tf.int32, [None, None], name = 'entity_ids')
    entity_raw_lens = tf.placeholder(tf.float32, [None], name = 'entity_raw_lens')
    l_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'l_context_ids')
    l_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'l_context_raw_lens')
    r_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'r_context_ids')
    r_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'r_context_raw_lens')

    word_emb = np.load('./data/word_emb.npy')
    word_emb_lookup_table = tf.Variable(word_emb, dtype=tf.float32, trainable = False, name = 'word_emb_lookup_table')

    l_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, l_context_ids)
    r_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, r_context_ids)
    entity_embs = tf.nn.embedding_lookup(word_emb_lookup_table, entity_ids)

    entity_lens = tf.reshape(entity_raw_lens, [-1, 1], name = 'entity_lens')
    l_context_lens = tf.reshape(l_context_raw_lens, [-1, 1], name = 'l_context_lens')
    r_context_lens = tf.reshape(r_context_raw_lens, [-1, 1], name = 'r_context_lens')

    drop_out = tf.placeholder(tf.float32)

    # l_context_embs_sum = tf.reduce_sum(l_context_embs, 1)
    # l_context_embs_ave = l_context_embs_sum / l_context_lens
    # r_context_embs_sum = tf.reduce_sum(r_context_embs, 1)
    # r_context_embs_ave = r_context_embs_sum / r_context_lens
    # context_embs_ave = tf.concat([l_context_embs_ave, r_context_embs_ave], 1, name = 'context_embs_ave')

    l_context_embs_sum = tf.reduce_sum(l_context_embs, 1)
    r_context_embs_sum = tf.reduce_sum(r_context_embs, 1)
    context_embs_sum = tf.concat([l_context_embs_sum, r_context_embs_sum], 1, name = 'context_embs_ave')

    entity_embs_sum = tf.reduce_sum(entity_embs, 1)
    entity_embs_ave = entity_embs_sum / entity_lens

    entity_embs_ave_dropout = tf.nn.dropout(entity_embs_ave, 1.0 - drop_out)

    # Xs = tf.concat([entity_embs_ave_dropout,context_embs_ave], 1)
    Xs = tf.concat([entity_embs_ave_dropout,context_embs_sum], 1)
    Ys = tf.placeholder(tf.float32, [None, 113], name = 'Ys')

    W = tf.Variable(tf.random_uniform([900, 113], minval=-0.01, maxval=0.01), name = 'W')
    logit = tf.matmul(Xs, W)

    predict_y = tf.sigmoid(logit)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Ys, logits = logit))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    figer = figer_data_multi_label_batcher.figer_data_multi_label()
    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

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
                                batch_r_context_ids, batch_r_context_lens, batch_ys = figer.next_batch()
                _, print_loss = \
                sess.run([train_step, loss],\
                 feed_dict={entity_ids: batch_entity_ids, entity_raw_lens: batch_entity_raw_lens, \
                            l_context_ids: batch_l_context_ids, l_context_raw_lens: batch_l_context_lens, \
                            r_context_ids: batch_r_context_ids, r_context_raw_lens: batch_r_context_lens, Ys: batch_ys, drop_out: [0.0]})

                with open('temp.txt', 'w') as f:
                    f.write('training epoch: {} batch {} : {}\n'.format(epoch, i, print_loss))

            figer_test.train_pos = 0
            batch_entity_ids, batch_entity_raw_lens, batch_l_context_ids, batch_l_context_lens, \
                            batch_r_context_ids, batch_r_context_lens, batch_ys = figer_test.next_batch()
            print_entity_embs_ave, print_predict_y = sess.run([entity_embs_ave, predict_y],\
             feed_dict={entity_ids: batch_entity_ids, entity_raw_lens: batch_entity_raw_lens, \
                        l_context_ids: batch_l_context_ids, l_context_raw_lens: batch_l_context_lens, \
                        r_context_ids: batch_r_context_ids, r_context_raw_lens: batch_r_context_lens, drop_out: [0.0]})

            evaluate.acc_hook(print_predict_y, batch_ys, epoch)


if __name__ == "__main__":
    state_of_the_art_average_emb()
    # w2v_type_linear()
    # temp()
