import tensorflow as tf
import numpy as np
import mix_data_batcher
from operator import itemgetter
import evaluate
from sklearn.externals import joblib
# import cPickle as pickle
import pickle
import argparse
import sys
import data_utility



def mix_seen_type_dot_distance_label_matrix():
    model_flag, feature_flag, entity_type_feature_flag, exact_entity_type_feature_flag, \
        type_only_feature_flag, id_select_flag, log_path, log_head= my_argparse()

    with open(log_path, 'w') as f:
        f.write('{}\n'.format(log_head))


    seen_label_ids = range(0, 1500)
    dev_unseen_label_ids = range(0, 1500)
    test_unseen_label_ids = range(1500, 1500+2988)

    seen_label_ids = np.sort(seen_label_ids)
    dev_unseen_label_ids = np.sort(dev_unseen_label_ids)
    test_unseen_label_ids = np.sort(test_unseen_label_ids)

    test_prior = np.full(test_unseen_label_ids.shape, 0.03)


    # test_prior = data_utility.get_ave_top_k_prior(test_unseen_label_ids, seen_label_ids, 3)


    with tf.variable_scope("foo"):
        train_placeholders, train_train_step, train_loss, train_predict_y = \
        create_model(1, seen_label_ids=seen_label_ids, model_flag=model_flag, feature_flag=feature_flag, \
                    entity_type_feature_flag=entity_type_feature_flag, exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)
        tf.get_variable_scope().reuse_variables()
        test_placeholders, _, test_loss, test_predict_y = \
        create_model(2, test_unseen_label_ids=test_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)

        dev_placeholders, _, dev_loss, dev_predict_y = \
        create_model(3, dev_unseen_label_ids=dev_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)


    print('before loading')
    figer = mix_data_batcher.mix_data_multi_label()
    figer_dev = mix_data_batcher.mix_data_multi_label()

    # figer_dev = mix_data_batcher.mix_data_multi_label()
    figer_test = mix_data_batcher.mix_data_multi_label(entity_file = 'mix_data/test_mix_word_with_context.txt', \
                    context_file = 'mix_data/test_mix_tagged_context.txt', \
                    type_file = 'mix_data/test_mix_Types_with_context_sparse.npz')

    print('finish loading')


    training_F1s = []
    dev_F1s = []
    test_F1s = []

    config = tf.ConfigProto(
        device_count = {'GPU': 0},
        intra_op_parallelism_threads=24,
        inter_op_parallelism_threads=24
    )
    type_f1_info = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, 5):
            figer.shuffle()
            epoch_type_f1_info = data_utility.type_f1s()
            for i in range(0, 2037):
            # for i in range(0, 1):
                batch_data = figer.next_batch(seen_label_ids)
                # if i in range(300, 370):
                #     continue
                feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.5]) + list([0.0])))

                _, print_loss = sess.run([train_train_step, train_loss], feed_dict)

                with open('temp.txt', 'w') as f:
                    f.write('training epoch: {} batch {} : {}\n'.format(epoch, i, print_loss))

            # # training performance
            # predict_ys = np.zeros([0, len(seen_label_ids)])
            # truth_ys = np.zeros([0, len(seen_label_ids)])
            # for i in range(0, 2037):
            # # for i in range(0, 1):
            #     batch_data = figer.next_batch(seen_label_ids)
            #     feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
            #     print_predict_y = sess.run(train_predict_y, feed_dict)
            #     predict_ys = np.vstack((predict_ys, print_predict_y))
            #     truth_ys = np.vstack((truth_ys, batch_data[-1]))
            #     with open('temp.txt', 'w') as f:
            #         f.write('testing train batch {} : {}\n'.format(i, print_loss))
            #
            # F1, train_type_f1s, train_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 1, log_path)
            # training_F1s.append(F1)
            # epoch_type_f1_info.add_f1s(seen_label_ids, train_type_f1s, train_F1_num, 0)
            #
            #
            # # dev performance
            # predict_ys = np.zeros([0, len(dev_unseen_label_ids)])
            # truth_ys = np.zeros([0, len(dev_unseen_label_ids)])
            # for i in range(0, 2037):
            # # for i in range(0, 1):
            #     batch_data = figer_dev.next_batch(dev_unseen_label_ids)
            #     feed_dict = dict(zip(dev_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
            #     print_predict_y = sess.run(dev_predict_y, feed_dict)
            #     predict_ys = np.vstack((predict_ys, print_predict_y))
            #     truth_ys = np.vstack((truth_ys, batch_data[-1]))
            #     with open('temp.txt', 'w') as f:
            #         f.write('testing dev batch {} : {}\n'.format(i, print_loss))
            # F1, dev_type_f1s, dev_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 2, log_path)
            # dev_F1s.append(F1)
            # epoch_type_f1_info.add_f1s(dev_unseen_label_ids, dev_type_f1s, dev_F1_num, 1)


            # test performance

            truth_ys = np.zeros(((7)*1000, len(test_unseen_label_ids)))
            predict_ys = np.zeros(((7)*1000, len(test_unseen_label_ids)))
            for i in range(0, 7):
                batch_data = figer_test.next_batch(test_unseen_label_ids)
                # if i in range(0, 300):
                #     continue
                feed_dict = dict(zip(test_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(test_predict_y, feed_dict)
                predict_ys[(i)*1000:(i+1)*1000] = print_predict_y
                # predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys[(i)*1000:(i+1)*1000] = batch_data[-1]
                # truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing test batch {} : {}\n'.format(i, print_loss))

            # flag_array = np.zeros(1387)
            #
            # for i in range(0, truth_ys.shape[0]):
            #     for j in range(0, 1387):
            #         if truth_ys[i][j] == 1:
            #             flag_array[j] = 1
            # print np.sum(flag_array)
            #
            # for i in range(0, 1387):
            #     if flag_array[i] == 0:
            #         print i
            # print 'done'
            # while True:
            #     pass
            #
            # for i in range(0, predict_ys.shape[0]):
            #     print predict_ys[i]
            #     print truth_ys[i]



            # F1, test_type_f1s, test_F1_num = evaluate.acc_hook_f1_break_down(predict_ys,truth_ys, epoch, 0, log_path, prior=test_prior)
            # test_F1s.append(F1)
            # epoch_type_f1_info.add_f1s(test_unseen_label_ids, test_type_f1s, test_F1_num, 2)
            epoch_type_f1_info.add_test_auc(test_unseen_label_ids, truth_ys, predict_ys)
            type_f1_info.append(epoch_type_f1_info)


        print('finish')

        # print(epoch_type_f1_info.test_auc_dict)



        # with open('./umls_type_f1_file/' + log_path[19:-4] + '.pickle', 'w') as outfile:
        #     pickle.dump(type_f1_info, outfile)


        # dev_max_id = np.argmax(dev_F1s, 0)[2][2]
        #
        # max_dev_test_micro = test_F1s[dev_max_id][2][2]
        # max_test_micro = np.amax(test_F1s, 0)[2][2]
        #
        # max_dev_test_macro = test_F1s[dev_max_id][1][2]
        # max_test_macro = np.amax(test_F1s, 0)[1][2]
        #
        # with open(log_path, 'a') as f:
        #     f.write('max__d_e_v__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_dev_test_macro, max_dev_test_micro))
        #     f.write('max__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_test_macro, max_test_micro))


        with open('result.pkl', 'wb') as f:
            pickle.dump(type_f1_info, f)


        with open(log_path, 'a') as f:
            f.write('test type ave_auc')
            for i in range(0, len(type_f1_info)):
                f.write('epoch= {} test_type_ave_auc= {}\n'.format(i, np.mean(list(type_f1_info[i].test_auc_dict.values()))))
                print(np.mean(list(epoch_type_f1_info.test_auc_dict.values())))



def create_model(select_flag, seen_label_ids=None, test_unseen_label_ids=None, dev_unseen_label_ids=None, model_flag = 0, \
feature_flag = 0, entity_type_feature_flag = 0, exact_entity_type_feature_flag = 0, type_only_feature_flag = 0):
    if model_flag == 0:
        rep_len = 900
    elif model_flag == 1 or model_flag == 2:
        rep_len = 500

    if feature_flag == 1:
        rep_len += 50

    window_size = 10
    entity_ids = tf.placeholder(tf.int32, [None, None], name = 'entity_ids')
    entity_raw_lens = tf.placeholder(tf.float32, [None], name = 'entity_raw_lens')
    l_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'l_context_ids')
    l_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'l_context_raw_lens')
    r_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'r_context_ids')
    r_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'r_context_raw_lens')
    entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    exact_entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    type_only_features = tf.placeholder(tf.float32, [None, None, 3])

    word_emb = np.load('./mix_data/word_emb.npy').astype(np.float32)
    word_emb_lookup_table = tf.get_variable(initializer=word_emb, dtype=tf.float32, trainable = False, name = 'word_emb_lookup_table')

    # with open('umls_data/umls_labelid2emb.pkl', 'r') as f:
    #     label_id2emb = pickle.load(f)
    label_id2emb = np.load('./mix_data/mix_labelid2emb.npy')
    if select_flag == 1:
        label_id2emb = np.take(label_id2emb, seen_label_ids, 0)
        label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'train_label_id2emb_matrix')
    elif select_flag == 2:
        label_id2emb = np.take(label_id2emb, test_unseen_label_ids, 0)
        label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'test_label_id2emb_matrix')
    elif select_flag == 3:
            label_id2emb = np.take(label_id2emb, dev_unseen_label_ids, 0)
            label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'dev_label_id2emb_matrix')

    l_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, l_context_ids)
    r_context_embs = tf.nn.embedding_lookup(word_emb_lookup_table, r_context_ids)
    entity_embs = tf.nn.embedding_lookup(word_emb_lookup_table, entity_ids)

    entity_lens = tf.reshape(entity_raw_lens, [-1, 1], name = 'entity_lens')
    l_context_lens = tf.reshape(l_context_raw_lens, [-1, 1], name = 'l_context_lens')
    r_context_lens = tf.reshape(r_context_raw_lens, [-1, 1], name = 'r_context_lens')

    # this dropout share with feature dropout
    mention_drop_out = tf.placeholder(tf.float32)
    label_drop_out = tf.placeholder(tf.float32)

    # l_context_embs_sum = tf.reduce_sum(l_context_embs, 1)
    # l_context_embs_ave = l_context_embs_sum / l_context_lens
    # r_context_embs_sum = tf.reduce_sum(r_context_embs, 1)
    # r_context_embs_ave = r_context_embs_sum / r_context_lens
    # context_embs_ave = tf.concat([l_context_embs_ave, r_context_embs_ave], 1, name = 'context_embs_ave')

    if model_flag == 0:
        l_context_embs_sum = tf.reduce_sum(l_context_embs, 1)
        r_context_embs_sum = tf.reduce_sum(r_context_embs, 1)
        context_embs_rep = tf.concat([l_context_embs_sum, r_context_embs_sum], -1, name = 'context_embs_rep')
    elif model_flag == 1:
        left_lstm  = tf.contrib.rnn.LSTMCell(100)
        right_lstm = tf.contrib.rnn.LSTMCell(100)
        with tf.variable_scope("rnn_left") as scope:
            left_rnn,_  = tf.nn.dynamic_rnn(left_lstm, l_context_embs, dtype=tf.float32)
        with tf.variable_scope("rnn_right") as scope:
            right_rnn,_ = tf.nn.dynamic_rnn(right_lstm, tf.reverse(r_context_embs, [1]), dtype=tf.float32)
        context_embs_rep = tf.concat([left_rnn[:,-1], right_rnn[:,-1]], -1, name = 'context_embs_rep')
    elif model_flag == 2:
        left_lstm_F  = tf.contrib.rnn.LSTMCell(100)
        right_lstm_F = tf.contrib.rnn.LSTMCell(100)
        left_lstm_B  = tf.contrib.rnn.LSTMCell(100)
        right_lstm_B = tf.contrib.rnn.LSTMCell(100)
        with tf.variable_scope("rnn_left") as scope:
            left_birnn, _ = tf.nn.bidirectional_dynamic_rnn(left_lstm_F, left_lstm_B, l_context_embs, dtype=tf.float32)
        with tf.variable_scope("rnn_right") as scope:
            right_birnn, _ = tf.nn.bidirectional_dynamic_rnn(right_lstm_F, right_lstm_B, tf.reverse(r_context_embs, [1]), dtype=tf.float32)

        left_birnn_cat = tf.concat([left_birnn[0], left_birnn[1]], -1)
        right_birnn_cat = tf.concat([right_birnn[0], right_birnn[1]], -1)
        birnn_cat = tf.concat([left_birnn_cat, right_birnn_cat], 1)
        context_embs_rep = attentive_sum(birnn_cat, 200, 100)

    entity_embs_sum = tf.reduce_sum(entity_embs, 1)
    entity_embs_ave = entity_embs_sum / entity_lens

    entity_embs_ave_dropout = tf.nn.dropout(entity_embs_ave, 1.0 - mention_drop_out)

    features = tf.placeholder(tf.int32,[None, 70])
    if feature_flag == 1:
        feature_embeddings = tf.get_variable(initializer=tf.random_uniform([600000, 50], minval=-0.01, maxval=0.01), name='feature_embeddings')
        feature_rep = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(feature_embeddings, features),1), 1.0 - mention_drop_out)
        Xs = tf.concat([entity_embs_ave_dropout,context_embs_rep, feature_rep], 1)
    else:
        Xs = tf.concat([entity_embs_ave_dropout,context_embs_rep], 1)


    if select_flag == 1:
        train_Ys = tf.placeholder(tf.float32, [None, len(seen_label_ids)], name = 'train_Ys')
    elif select_flag == 2:
        test_Ys = tf.placeholder(tf.float32, [None, len(test_unseen_label_ids)], name = 'test_Ys')
    elif select_flag == 3:
        dev_Ys = tf.placeholder(tf.float32, [None, len(dev_unseen_label_ids)], name = 'dev_Ys')

    W = tf.get_variable(initializer=tf.random_uniform([rep_len, 300], minval=-0.01, maxval=0.01), name='W')
    input_representation = tf.matmul(Xs, W)

    # zero input_representation
    # input_representation = tf.zeros(tf.shape(input_representation), dtype=tf.float32)

    label_id2emb_matrix_dropout = tf.nn.dropout(label_id2emb_matrix, 1.0 - label_drop_out)

    logit = tf.matmul(input_representation, label_id2emb_matrix_dropout, transpose_b = True, name='logit')

    logit_expand = tf.expand_dims(logit, axis=-1)
    w_length = 1
    if entity_type_feature_flag == 1:
        w_length += 3
        logit_expand = tf.concat([logit_expand, entity_type_features], -1)
    if exact_entity_type_feature_flag == 1:
        w_length += 3
        logit_expand = tf.concat([logit_expand, exact_entity_type_features], -1)
    if type_only_feature_flag == 1:
        w_length += 3
        logit_expand = tf.concat([logit_expand, type_only_features], -1)

    final_W = tf.get_variable(initializer=tf.random_uniform([w_length, 1], minval=-0.01, maxval=0.01), name='final_W')
    logit = tf.squeeze(tf.layers.dense(logit_expand, 1, name='logit'), axis=-1)


    predict_y = tf.sigmoid(logit)
    # predict_y = tf.Print(predict_y, [predict_y])

    if select_flag == 1:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = train_Ys, logits = logit))
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss, name = 'train_step')
    elif select_flag == 2:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = test_Ys, logits = logit))
        train_step = tf.no_op()
    elif select_flag == 3:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = dev_Ys, logits = logit))
        train_step = tf.no_op()


    if select_flag == 1:
        return (entity_ids, entity_raw_lens,
                   l_context_ids, l_context_raw_lens,
                   r_context_ids, r_context_raw_lens, features, entity_type_features, \
                   exact_entity_type_features, type_only_features, train_Ys, mention_drop_out, label_drop_out), \
                   train_step, loss, predict_y
    elif select_flag == 2:
        return (entity_ids, entity_raw_lens,
                   l_context_ids, l_context_raw_lens,
                   r_context_ids, r_context_raw_lens, features, entity_type_features, \
                   exact_entity_type_features, type_only_features, test_Ys, mention_drop_out, label_drop_out), \
                   train_step, loss, predict_y
    elif select_flag == 3:
        return (entity_ids, entity_raw_lens,
                   l_context_ids, l_context_raw_lens,
                   r_context_ids, r_context_raw_lens, features, entity_type_features, \
                   exact_entity_type_features, type_only_features, dev_Ys, mention_drop_out, label_drop_out), \
                   train_step, loss, predict_y


def attentive_sum(inputs,input_dim, hidden_dim):
    with tf.variable_scope("attention"):
        W = tf.get_variable(initializer=tf.random_uniform([input_dim, hidden_dim], minval=-0.01, maxval=0.01), name='W')
        U = tf.get_variable(initializer=tf.random_uniform([hidden_dim, 1], minval=-0.01, maxval=0.01), name='U')
        dense_1 = tf.layers.dense(inputs, hidden_dim, use_bias=False)
        dense_2 = tf.layers.dense(dense_1, 1, use_bias=False)
        attentions = tf.nn.softmax(dense_2)
        weighted_inputs = tf.multiply(inputs, attentions)
        output = tf.reduce_sum(weighted_inputs, axis = 1)
    return output


def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_flag', help='model to train', choices=['ave','LSTM', 'attention'])
    parser.add_argument('feature_flag', help='figer feature flag', choices=[0], type=int)
    parser.add_argument('entity_type_feature_flag', help='entity type (general) feature flag', choices=[0], type=int)
    parser.add_argument('exact_entity_type_feature_flag', help='entity type (exact) feature flag', choices=[0], type=int)
    parser.add_argument('type_only_feature_flag', help='type only feature', choices=[0], type=int)
    parser.add_argument('id_select_flag', help='seen & unseen type select', choices=[0], type=int)
    parser.add_argument('-auto_gen_log_path', help='if auto gen log_path', choices=[1, 0], default=0, type=int)
    parser.add_argument('-log_path', help='path of the log file', default='loss_record.txt')

    args = parser.parse_args()

    if args.model_flag == 'ave':
        model_flag = 0
    elif args.model_flag == 'LSTM':
        model_flag = 1
    elif args.model_flag == 'attention':
        model_flag = 2

    feature_flag = args.feature_flag
    entity_type_feature_flag = args.entity_type_feature_flag
    exact_entity_type_feature_flag = args.exact_entity_type_feature_flag
    type_only_feature_flag = args.type_only_feature_flag
    id_select_flag = args.id_select_flag

    log_path = args.log_path
    log_head = ' '.join(sys.argv[1:])
    log_head = 'args = ' + log_head

    if args.auto_gen_log_path == 1:
        log_path = './mix_log_files/'
        log_path += (args.model_flag + '_')
        log_path += (str(args.feature_flag) + '_')
        log_path += (str(args.entity_type_feature_flag) + '_')
        log_path += (str(args.exact_entity_type_feature_flag) + '_')
        log_path += (str(args.type_only_feature_flag) + '_')
        log_path += (str(args.id_select_flag))
        log_path += '.txt'

    return model_flag, feature_flag, entity_type_feature_flag, exact_entity_type_feature_flag, \
        type_only_feature_flag, id_select_flag, log_path, log_head


def get_CV_info(file_name):

    n = 0
    train_ids = []
    dev_ids = []
    test_ids = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace('\n','')
            if n % 3 == 0:
                train_ids.append([int(e) for e in line.split(' ')])
            elif n % 3 == 1:
                dev_ids.append([int(e) for e in line.split(' ')])
            elif n % 3 == 2:
                test_ids.append([int(e) for e in line.split(' ')])
            n += 1

    return train_ids, dev_ids, test_ids


def temp():
    with open('result.pkl', 'rb') as f:
        type_f1_info = pickle.load(f)

    for i in range(0, 5):
        np.mean(list(type_f1_info[i].test_auc_dict.values()))



if __name__ == "__main__":
    mix_seen_type_dot_distance_label_matrix()
    # w2v_type_linear()
    # temp()
