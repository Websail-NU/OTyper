import numpy as np
import figer_data_multi_label_batcher
import umls_data_batcher
import umls_data_batcher
import evaluate
import tensorflow as tf
from operator import itemgetter
from sklearn.externals import joblib
import pickle
import argparse
import sys
import data_utility

def emb_baseline():
    model_flag, feature_flag, entity_type_feature_flag, exact_entity_type_feature_flag, \
        type_only_feature_flag, id_select_flag, log_path, log_head= my_argparse()

    with open(log_path, 'w') as f:
        f.write('{}\n'.format(log_head))

# 106/7
    if id_select_flag == 0:
        test_unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
        seen_label_ids = []
        dev_unseen_label_ids = []
        for i in range(0,113):
            if not i in test_unseen_label_ids:
                seen_label_ids.append(i)
                dev_unseen_label_ids.append(i)
    elif id_select_flag == 1:
        test_unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]
        seen_label_ids = []
        dev_unseen_label_ids = []
        for i in range(0,113):
            if not i in test_unseen_label_ids:
                seen_label_ids.append(i)
                dev_unseen_label_ids.append(i)
# seen all types
    elif id_select_flag == 2:
        seen_label_ids = range(0, 113)
        dev_unseen_label_ids = range(0, 113)
        test_unseen_label_ids = range(0, 113)
# CV
    elif id_select_flag >= 10 and id_select_flag < 20:
        train_ids, dev_ids, test_ids = get_CV_info('CV_output.txt')
        cv_id = id_select_flag % 10

        seen_label_ids = train_ids[cv_id]
        dev_unseen_label_ids = dev_ids[cv_id]
        test_unseen_label_ids = test_ids[cv_id]
    elif id_select_flag >= 20 and id_select_flag < 30:
        train_ids, dev_ids, test_ids = get_CV_info('CV_output.txt')
        cv_id = id_select_flag % 10

        seen_label_ids = train_ids[cv_id]
        dev_unseen_label_ids = seen_label_ids
        test_unseen_label_ids = test_ids[cv_id]

    seen_label_ids = np.sort(seen_label_ids)
    dev_unseen_label_ids = np.sort(dev_unseen_label_ids)
    test_unseen_label_ids = np.sort(test_unseen_label_ids)

    test_prior = np.full(test_unseen_label_ids.shape, 0.03)


# 99/7/7
    # if id_select_flag == 0:
    #     test_unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
    #     dev_unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]
    # elif id_select_flag == 1:
    #     test_unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]
    #     dev_unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
    #
    # seen_label_ids = []
    # for i in range(0,113):
    #     if (not i in test_unseen_label_ids) and (not i in dev_unseen_label_ids):
    #         seen_label_ids.append(i)

# original
    # test_unseen_label_ids = [76, 20, 18, 13, 2, 8, 3]
    # dev_unseen_label_ids = [76, 20, 18, 13, 2, 8, 3]
    # seen_label_ids = []
    #
    # for i in range(0,113):
    #     if not i in test_unseen_label_ids:
    #         seen_label_ids.append(i)


    with tf.variable_scope("foo"):
        train_placeholders, train_train_step, train_loss, train_predict_y = \
        create_model(1, 0, seen_label_ids=seen_label_ids, model_flag=model_flag, feature_flag=feature_flag, \
                    entity_type_feature_flag=entity_type_feature_flag, exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)
        tf.get_variable_scope().reuse_variables()
        test_placeholders, _, test_loss, test_predict_y = \
        create_model(2, 0, test_unseen_label_ids=test_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)

        dev_placeholders, _, dev_loss, dev_predict_y = \
        create_model(3, 0, dev_unseen_label_ids=dev_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)


    figer = figer_data_multi_label_batcher.figer_data_multi_label()
    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_test_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_test_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_test_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

    figer_dev = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_dev_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_dev_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_dev_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_dev_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_dev_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_dev_Types_with_context.npy')


    training_F1s = []
    dev_F1s = []
    test_F1s = []

    config = tf.ConfigProto(
        device_count = {'GPU': 0},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4
    )
    type_f1_info = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, 5):
            figer.shuffle()
            epoch_type_f1_info = data_utility.type_f1s()
            for i in range(0, 2000):
                batch_data = figer.next_batch(seen_label_ids)

                feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.5]) + list([0.0])))

                _, print_loss = sess.run([train_train_step, train_loss], feed_dict)

                with open('temp.txt', 'w') as f:
                    f.write('training epoch: {} batch {} : {}\n'.format(epoch, i, print_loss))

            # training performance
            predict_ys = np.zeros([0, len(seen_label_ids)])
            truth_ys = np.zeros([0, len(seen_label_ids)])
            for i in range(0, 200):
                batch_data = figer.next_batch(seen_label_ids)
                feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(train_predict_y, feed_dict)
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing train batch {} : {}\n'.format(i, print_loss))

            F1, train_type_f1s, train_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 1, log_path)
            training_F1s.append(F1)
            epoch_type_f1_info.add_f1s(seen_label_ids, train_type_f1s, train_F1_num, 0)


            # dev performance
            predict_ys = np.zeros([0, len(dev_unseen_label_ids)])
            truth_ys = np.zeros([0, len(dev_unseen_label_ids)])
            for i in range(0, 10):
                batch_data = figer_dev.next_batch(dev_unseen_label_ids)
                feed_dict = dict(zip(dev_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(dev_predict_y, feed_dict)
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing dev batch {} : {}\n'.format(i, print_loss))
            F1, dev_type_f1s, dev_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 2, log_path)
            dev_F1s.append(F1)
            epoch_type_f1_info.add_f1s(dev_unseen_label_ids, dev_type_f1s, dev_F1_num, 1)


            # test performance
            batch_data = figer_test.next_batch(test_unseen_label_ids)
            feed_dict = dict(zip(test_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
            print_predict_y = sess.run(test_predict_y, feed_dict)
            np.save('temp_predict', print_predict_y)
            np.save('temp_truth', batch_data[-1])
            F1, test_type_f1s, test_F1_num = evaluate.acc_hook_f1_break_down(print_predict_y, batch_data[-1], epoch, 0, log_path, prior=test_prior)
            test_F1s.append(F1)
            epoch_type_f1_info.add_f1s(test_unseen_label_ids, test_type_f1s, test_F1_num, 2)
            epoch_type_f1_info.add_test_auc(test_unseen_label_ids, batch_data[-1], print_predict_y)

            type_f1_info.append(epoch_type_f1_info)


        if log_path != 'loss_record.txt':
            print ('./base_line_type_f1_file/' + log_path[22:-4] + '.pickle')
            with open('./base_line_type_f1_file/' + log_path[22:-4] + '.pickle', 'w') as outfile:
                pickle.dump(type_f1_info, outfile)


        # dev_ave = np.average(dev_F1s, 1)
        # dev_F1s = np.asarray(dev_F1s, dtype=np.float32)
        # print dev_F1s.shape
        # print dev_ave.shape
        # while True:
        #     pass
        dev_max_id = np.argmax(dev_F1s, 0)[2][2]

        max_dev_test_micro = test_F1s[dev_max_id][2][2]
        max_test_micro = np.amax(test_F1s, 0)[2][2]

        max_dev_test_macro = test_F1s[dev_max_id][1][2]
        max_test_macro = np.amax(test_F1s, 0)[1][2]

        with open(log_path, 'a') as f:
            f.write('max__d_e_v__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_dev_test_macro, max_dev_test_micro))
            f.write('max__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_test_macro, max_test_micro))


def emb_baseline_umls():
    model_flag, feature_flag, entity_type_feature_flag, exact_entity_type_feature_flag, \
        type_only_feature_flag, id_select_flag, log_path, log_head= my_argparse()

    with open(log_path, 'w') as f:
        f.write('{}\n'.format(log_head))


    if id_select_flag == 0:
        seen_label_ids = list(range(0, 800))
        dev_unseen_label_ids = list(range(0, 800))
        test_unseen_label_ids = list(range(800, 1387))
    elif id_select_flag >= 10 and id_select_flag < 20:
        train_ids, dev_ids, test_ids = get_CV_info('CV_output_MSH.txt')
        cv_id = id_select_flag % 10

        seen_label_ids = train_ids[cv_id]
        dev_unseen_label_ids = dev_ids[cv_id]
        test_unseen_label_ids = test_ids[cv_id]


    seen_label_ids = np.sort(seen_label_ids)
    dev_unseen_label_ids = np.sort(dev_unseen_label_ids)
    test_unseen_label_ids = np.sort(test_unseen_label_ids)

    test_prior = np.full(test_unseen_label_ids.shape, 0.03)


# 99/7/7
    # if id_select_flag == 0:
    #     test_unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
    #     dev_unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]
    # elif id_select_flag == 1:
    #     test_unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]
    #     dev_unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
    #
    # seen_label_ids = []
    # for i in range(0,113):
    #     if (not i in test_unseen_label_ids) and (not i in dev_unseen_label_ids):
    #         seen_label_ids.append(i)

# original
    # test_unseen_label_ids = [76, 20, 18, 13, 2, 8, 3]
    # dev_unseen_label_ids = [76, 20, 18, 13, 2, 8, 3]
    # seen_label_ids = []
    #
    # for i in range(0,113):
    #     if not i in test_unseen_label_ids:
    #         seen_label_ids.append(i)


    with tf.variable_scope("foo"):
        train_placeholders, train_train_step, train_loss, train_predict_y = \
        create_model(1, 1, seen_label_ids=seen_label_ids, model_flag=model_flag, feature_flag=feature_flag, \
                    entity_type_feature_flag=entity_type_feature_flag, exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)
        tf.get_variable_scope().reuse_variables()
        test_placeholders, _, test_loss, test_predict_y = \
        create_model(2, 1, test_unseen_label_ids=test_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)

        dev_placeholders, _, dev_loss, dev_predict_y = \
        create_model(3, 1, dev_unseen_label_ids=dev_unseen_label_ids, model_flag=model_flag, \
                    feature_flag=feature_flag, entity_type_feature_flag=entity_type_feature_flag, \
                    exact_entity_type_feature_flag=exact_entity_type_feature_flag, \
                    type_only_feature_flag=type_only_feature_flag)


    umls_train = umls_data_batcher.umls_data_multi_label(entity_file = 'umls_data/train_refined_umls_word.txt', \
                    context_file = 'umls_data/train_refined_umls_tagged_context.txt', \
                    entity_type_exact_feature_file = 'umls_data/train_umls_exact_et_features.npy', \
                    type_file = 'umls_data/train_umls_Types_with_context.npy')
    umls_test = umls_data_batcher.umls_data_multi_label(entity_file = 'umls_data/test_refined_umls_word.txt', \
                    context_file = 'umls_data/test_refined_umls_tagged_context.txt', \
                    entity_type_exact_feature_file = 'umls_data/test_umls_exact_et_features.npy', \
                    type_file = 'umls_data/test_umls_Types_with_context.npy')
    umls_dev = umls_data_batcher.umls_data_multi_label(entity_file = 'umls_data/dev_refined_umls_word.txt', \
                    context_file = 'umls_data/dev_refined_umls_tagged_context.txt', \
                    entity_type_exact_feature_file = 'umls_data/dev_umls_exact_et_features.npy', \
                    type_file = 'umls_data/dev_umls_Types_with_context.npy')


    training_F1s = []
    dev_F1s = []
    test_F1s = []

    config = tf.ConfigProto(
        device_count = {'GPU': 0},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4
    )
    type_f1_info = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, 5):
            umls_train.shuffle()
            epoch_type_f1_info = data_utility.type_f1s()
            for i in range(0, 296):
            # for i in range(0, 2):
                batch_data = umls_train.next_batch(seen_label_ids)

                feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.5]) + list([0.0])))

                _, print_loss = sess.run([train_train_step, train_loss], feed_dict)

                with open('temp.txt', 'w') as f:
                    f.write('training epoch: {} batch {} : {}\n'.format(epoch, i, print_loss))

            # training performance
            predict_ys = np.zeros([0, len(seen_label_ids)])
            truth_ys = np.zeros([0, len(seen_label_ids)])
            for i in range(0, 296):
            # for i in range(0, 2):
                batch_data = umls_train.next_batch(seen_label_ids)
                feed_dict = dict(zip(train_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(train_predict_y, feed_dict)
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing train batch {} : {}\n'.format(i, print_loss))

            F1, train_type_f1s, train_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 1, log_path)
            training_F1s.append(F1)
            epoch_type_f1_info.add_f1s(seen_label_ids, train_type_f1s, train_F1_num, 0)


            # dev performance
            predict_ys = np.zeros([0, len(dev_unseen_label_ids)])
            truth_ys = np.zeros([0, len(dev_unseen_label_ids)])
            for i in range(0, 37):
            # for i in range(0, 3):
                batch_data = umls_dev.next_batch(dev_unseen_label_ids)
                feed_dict = dict(zip(dev_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(dev_predict_y, feed_dict)
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing dev batch {} : {}\n'.format(i, print_loss))
            F1, dev_type_f1s, dev_F1_num = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, epoch, 2, log_path)
            dev_F1s.append(F1)
            epoch_type_f1_info.add_f1s(dev_unseen_label_ids, dev_type_f1s, dev_F1_num, 1)


            # test performance
            predict_ys = np.zeros([0, len(test_unseen_label_ids)])
            truth_ys = np.zeros([0, len(test_unseen_label_ids)])
            for i in range(0, 37):
            # for i in range(0, 3):
                batch_data = umls_test.next_batch(test_unseen_label_ids)
                feed_dict = dict(zip(test_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
                print_predict_y = sess.run(test_predict_y, feed_dict)
                predict_ys = np.vstack((predict_ys, print_predict_y))
                truth_ys = np.vstack((truth_ys, batch_data[-1]))
                with open('temp.txt', 'w') as f:
                    f.write('testing test batch {} : {}\n'.format(i, print_loss))
            F1, test_type_f1s, test_F1_num = evaluate.acc_hook_f1_break_down(print_predict_y, batch_data[-1], epoch, 0, log_path, prior=test_prior)
            test_F1s.append(F1)
            epoch_type_f1_info.add_f1s(test_unseen_label_ids, test_type_f1s, test_F1_num, 2)
            epoch_type_f1_info.add_test_auc(test_unseen_label_ids, batch_data[-1], print_predict_y)


            # batch_data = umls_test.next_batch(test_unseen_label_ids)
            # feed_dict = dict(zip(test_placeholders, list(batch_data) + list([0.0]) + list([0.0])))
            # print_predict_y = sess.run(test_predict_y, feed_dict)
            # F1, test_type_f1s, test_F1_num = evaluate.acc_hook_f1_break_down(print_predict_y, batch_data[-1], epoch, 0, log_path, prior=test_prior)
            # test_F1s.append(F1)
            # epoch_type_f1_info.add_f1s(test_unseen_label_ids, test_type_f1s, test_F1_num, 2)
            # epoch_type_f1_info.add_test_auc(test_unseen_label_ids, batch_data[-1], print_predict_y)

            type_f1_info.append(epoch_type_f1_info)


        if log_path != 'loss_record.txt':
            print ('./base_line_type_f1_file/' + log_path[22:-4] + '.pickle')
            with open('./base_line_type_f1_file/' + log_path[22:-4] + '.pickle', 'wb') as outfile:
                pickle.dump(type_f1_info, outfile)


        # dev_ave = np.average(dev_F1s, 1)
        # dev_F1s = np.asarray(dev_F1s, dtype=np.float32)
        # print dev_F1s.shape
        # print dev_ave.shape
        # while True:
        #     pass
        dev_max_id = np.argmax(dev_F1s, 0)[2][2]

        max_dev_test_micro = test_F1s[dev_max_id][2][2]
        max_test_micro = np.amax(test_F1s, 0)[2][2]

        max_dev_test_macro = test_F1s[dev_max_id][1][2]
        max_test_macro = np.amax(test_F1s, 0)[1][2]

        with open(log_path, 'a') as f:
            f.write('max__d_e_v__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_dev_test_macro, max_dev_test_micro))
            f.write('max__t_e_s_t__macro__micro= {:.4f}\t{:.4f}\n'.format(max_test_macro, max_test_micro))


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


def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_flag', help='model to train', choices=['emb_cat', 'emb_sub'])
    parser.add_argument('feature_flag', help='figer feature flag', choices=[0,1], type=int)
    parser.add_argument('entity_type_feature_flag', help='entity type (general) feature flag', choices=[0], type=int)
    parser.add_argument('exact_entity_type_feature_flag', help='entity type (exact) feature flag', choices=[0,1], type=int)
    parser.add_argument('type_only_feature_flag', help='type only feature', choices=[0], type=int)
    parser.add_argument('id_select_flag', help='seen & unseen type select', choices=list(range(0,3))+list(range(10,30)), type=int)
    parser.add_argument('-auto_gen_log_path', help='if auto gen log_path', choices=[1,0], default=0, type=int)
    parser.add_argument('-log_path', help='path of the log file', default='loss_record.txt')

    args = parser.parse_args()

    if args.model_flag == 'emb_cat':
        model_flag = 11
    elif args.model_flag == 'emb_sub':
        model_flag = 12

    feature_flag = args.feature_flag
    entity_type_feature_flag = args.entity_type_feature_flag
    exact_entity_type_feature_flag = args.exact_entity_type_feature_flag
    type_only_feature_flag = args.type_only_feature_flag
    id_select_flag = args.id_select_flag

    log_path = args.log_path
    log_head = ' '.join(sys.argv[1:])
    log_head = 'args = ' + log_head

    if args.auto_gen_log_path == 1:
        log_path = './base_line_log_files/'
        log_path += (args.model_flag + '_')
        log_path += (str(args.feature_flag) + '_')
        log_path += (str(args.entity_type_feature_flag) + '_')
        log_path += (str(args.exact_entity_type_feature_flag) + '_')
        log_path += (str(args.type_only_feature_flag) + '_')
        log_path += (str(args.id_select_flag))
        log_path += '.txt'

    return model_flag, feature_flag, entity_type_feature_flag, exact_entity_type_feature_flag, \
        type_only_feature_flag, id_select_flag, log_path, log_head


def create_model(select_flag, data_flag, seen_label_ids=None, test_unseen_label_ids=None, dev_unseen_label_ids=None, model_flag = 0, \
feature_flag = 0, entity_type_feature_flag = 0, exact_entity_type_feature_flag = 0, type_only_feature_flag = 0):

    window_size = 10

    l_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'l_context_ids')
    l_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'l_context_raw_lens')
    r_context_ids = tf.placeholder(tf.int32, [None, window_size], name = 'r_context_ids')
    r_context_raw_lens = tf.placeholder(tf.float32, [None], name = 'r_context_raw_lens')
    entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    exact_entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    type_only_features = tf.placeholder(tf.float32, [None, None, 3])

    entity_ids = tf.placeholder(tf.int32, [None, None], name = 'entity_ids')
    entity_raw_lens = tf.placeholder(tf.float32, [None], name = 'entity_raw_lens')

    entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    exact_entity_type_features = tf.placeholder(tf.float32, [None, None, 3])
    type_only_features = tf.placeholder(tf.float32, [None, None, 3])

    if data_flag == 0:
        word_emb = np.load('./data/word_emb.npy').astype(np.float32)
    else:
        word_emb = np.load('umls_data/umls_word_emb.npy').astype(np.float32)

    word_emb_lookup_table = tf.get_variable(initializer=word_emb, dtype=tf.float32, trainable = False, name = 'word_emb_lookup_table')

    if data_flag == 0:
        label_id2emb = np.load('data/labelid2emb.npy')
    else:
        label_id2emb = np.load('umls_data/umls_labelid2emb.npy')

    if select_flag == 1:
        label_id2emb = np.take(label_id2emb, seen_label_ids, 0)
        label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'train_label_id2emb_matrix')
    elif select_flag == 2:
        label_id2emb = np.take(label_id2emb, test_unseen_label_ids, 0)
        label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'test_label_id2emb_matrix')
    elif select_flag == 3:
        label_id2emb = np.take(label_id2emb, dev_unseen_label_ids, 0)
        label_id2emb_matrix = tf.constant(label_id2emb, dtype=tf.float32, name = 'dev_label_id2emb_matrix')

    entity_embs = tf.nn.embedding_lookup(word_emb_lookup_table, entity_ids)
    entity_lens = tf.reshape(entity_raw_lens, [-1, 1], name = 'entity_lens')

    # this dropout share with feature dropout
    mention_drop_out = tf.placeholder(tf.float32)
    label_drop_out = tf.placeholder(tf.float32)


    entity_embs_sum = tf.reduce_sum(entity_embs, 1)
    entity_embs_ave = entity_embs_sum / entity_lens

    entity_embs_ave_dropout = tf.nn.dropout(entity_embs_ave, 1.0 - mention_drop_out)

    if select_flag == 1:
        train_Ys = tf.placeholder(tf.float32, [None, len(seen_label_ids)], name = 'train_Ys')
    elif select_flag == 2:
        test_Ys = tf.placeholder(tf.float32, [None, len(test_unseen_label_ids)], name = 'test_Ys')
    elif select_flag == 3:
        dev_Ys = tf.placeholder(tf.float32, [None, len(dev_unseen_label_ids)], name = 'dev_Ys')


    features = tf.placeholder(tf.int32,[None, 70])
    feature_embeddings = tf.get_variable(initializer=tf.random_uniform([600000, 50], minval=-0.01, maxval=0.01), name='feature_embeddings')
    feature_rep = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(feature_embeddings, features),1), 1.0 - mention_drop_out)
    entity_embs_ave_dropout = tf.expand_dims(entity_embs_ave_dropout, 1)
    label_id2emb_matrix = tf.expand_dims(label_id2emb_matrix, 0)
    label_id2emb_matrix = tf.tile(label_id2emb_matrix, [tf.shape(entity_embs_ave_dropout)[0],1,1])
    feature_rep = tf.expand_dims(feature_rep, 1)

    if select_flag == 1:
        type_num = len(seen_label_ids)
    elif select_flag == 2:
        type_num = len(test_unseen_label_ids)
    elif select_flag == 3:
        type_num = len(dev_unseen_label_ids)

    entity_embs_ave_dropout = tf.tile(entity_embs_ave_dropout, [1, type_num, 1])
    feature_rep = tf.tile(feature_rep, [1, type_num, 1])

    if model_flag == 11:
        if feature_flag == 1:
            rep = tf.concat([entity_embs_ave_dropout, label_id2emb_matrix, feature_rep], -1)
        else:
            rep = tf.concat([entity_embs_ave_dropout, label_id2emb_matrix], -1)
    elif model_flag == 12:
        if feature_flag == 1:
            rep = tf.concat([entity_embs_ave_dropout-label_id2emb_matrix, feature_rep], -1)
        else:
            rep = tf.concat([entity_embs_ave_dropout-label_id2emb_matrix], -1)

    last_dim = int(rep.get_shape()[-1])
    rep = tf.reshape(rep, [-1, last_dim])


    with tf.variable_scope('last_layer'):
        logit = tf.layers.dense(rep, 1)
        if exact_entity_type_feature_flag == 1:
            exact_entity_type_features_modified = tf.reshape(exact_entity_type_features, [-1, 3])
            logit = tf.concat([logit, exact_entity_type_features_modified], -1)
            # last_W = tf.get_variable(initializer=tf.random_uniform([4, 1], minval=-0.01, maxval=0.01), name='last_W')
            logit = tf.layers.dense(logit, 1)


    if model_flag == 11:
        predict_y = tf.reshape(tf.sigmoid(logit), [-1, type_num])
    elif model_flag == 12:
        predict_y = tf.reshape(tf.sigmoid(logit), [-1, type_num])

    if select_flag == 1:
        # train_Ys = tf.reshape(train_Ys, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.reshape(train_Ys, [-1, 1]), logits = logit))
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss, name = 'train_step')
    elif select_flag == 2:
        # test_Ys = tf.reshape(test_Ys, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.reshape(test_Ys, [-1, 1]), logits = logit))
        train_step = tf.no_op()
    elif select_flag == 3:
        # dev_Ys = tf.reshape(dev_Ys, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.reshape(dev_Ys, [-1, 1]), logits = logit))
        train_step = tf.no_op()


    # if select_flag == 1:
    #     return (entity_ids, entity_raw_lens, features, \
    #                train_Ys, mention_drop_out, label_drop_out), \
    #                train_step, loss, predict_y
    # elif select_flag == 2:
    #     return (entity_ids, entity_raw_lens, features, \
    #                test_Ys, mention_drop_out, label_drop_out), \
    #                train_step, loss, predict_y
    #     # return (entity_ids, entity_raw_lens,
    #     #            l_context_ids, l_context_raw_lens,
    #     #            r_context_ids, r_context_raw_lens, features, entity_type_features, \
    #     #            exact_entity_type_features, type_only_features, test_Ys, mention_drop_out, label_drop_out), \
    #     #            train_step, loss, predict_y
    # elif select_flag == 3:
    #     return (entity_ids, entity_raw_lens, features, \
    #                dev_Ys, mention_drop_out, label_drop_out), \
    #                train_step, loss, predict_y

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


def pattern_baseline():
    # unseen_label_ids = [76, 20, 18, 13, 2, 8, 3]
    # unseen_label_ids = [1, 0, 11, 76, 18, 13, 9]
    # unseen_label_ids = [4, 13, 38, 9]
    # unseen_label_ids = [2, 3, 8, 20, 5, 24, 38]

    all_unseen_label_ids = []
    for id_select_flag in range(0, 10):
        train_ids, dev_ids, test_ids = get_CV_info('CV_output.txt')
        cv_id = id_select_flag % 10

        for e in test_ids[cv_id]:
            all_unseen_label_ids.append(e)

    unseen_label_ids = all_unseen_label_ids

    unseen_label_ids = np.sort(unseen_label_ids)

    figer_test = figer_data_multi_label_batcher.figer_data_multi_label(entity_file = 'data/state_of_the_art_test_word_with_context.txt', \
                    context_file = 'data/state_of_the_art_test_tagged_context.txt', \
                    feature_file = 'data/state_of_the_art_test_Feature.npy', \
                    entity_type_feature_file = 'data/state_of_the_art_test_et_features.npy',\
                    entity_type_exact_feature_file = 'data/state_of_the_art_test_exact_et_features.npy',\
                    type_file = 'data/state_of_the_art_test_Types_with_context.npy')

    predict_ys = np.zeros([0, len(unseen_label_ids)])
    truth_ys = np.zeros([0, len(unseen_label_ids)])
    for i in range(0, 1):
        batch_data = figer_test.next_batch(unseen_label_ids)
        print_predict_y = baseline_predict(batch_data[-3])
        predict_ys = np.vstack((predict_ys, print_predict_y))
        truth_ys = np.vstack((truth_ys, batch_data[-1]))

    print(predict_ys.shape)
    print(truth_ys.shape)

    F1, test_type_f1s, _ = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, 0, 0, 'baseline.txt')

    with open('data/test_type_count.pkl', 'r') as f:
        b = pickle.load(f)

    count_sum = 0
    for e in unseen_label_ids:
        count_sum += b[e][1]

    F1_weighted = 0.0
    p = 0
    all_test_id = data_utility.get_all_unseen_types()
    for i in all_test_id:
        if i in unseen_label_ids:
            F1_weighted += (b[i][1]/count_sum) * test_type_f1s[p]
            p += 1
    print('pattern baseline type averaged macro = {}'.format(F1_weighted))
    print('pattern baseline type micro = {}'.format(F1[2][2]))

    epoch_type_f1_info = data_utility.type_f1s()
    epoch_type_f1_info.add_test_auc(unseen_label_ids, batch_data[-1], print_predict_y)

    #
    # print epoch_type_f1_info.test_auc_dict
    auc_array = []
    for k in epoch_type_f1_info.test_auc_dict:
        auc_array.append(epoch_type_f1_info.test_auc_dict[k])

    print('auc ave = {}'.format(np.average(auc_array)))



def pattern_baseline_umls():
    unseen_label_ids = range(1387-587,1387)

    umls_test = umls_data_batcher.umls_data_multi_label()

    predict_ys = np.zeros([69*100, len(unseen_label_ids)])
    truth_ys = np.zeros([69*100, len(unseen_label_ids)])
    for i in range(0, 369):
        if i < 300:
            continue
        print(i)
        batch_data = umls_test.next_batch(unseen_label_ids)
        print_predict_y = baseline_predict(batch_data[-3])
        predict_ys[(i-300)*100:(i+1-300)*100] = print_predict_y
        truth_ys[(i-300)*100:(i+1-300)*100] = batch_data[-1]
        # predict_ys = np.vstack((predict_ys, print_predict_y))
        # truth_ys = np.vstack((truth_ys, batch_data[-1]))

    print(predict_ys.shape)
    print(truth_ys.shape)

    # F1, test_type_f1s, _ = evaluate.acc_hook_f1_break_down(predict_ys, truth_ys, 0, 0, 'baseline.txt')

    # with open('data/test_type_count.pkl', 'r') as f:
    #     b = pickle.load(f)
    #
    # count_sum = 0
    # for e in unseen_label_ids:
    #     count_sum += b[e][1]
    #
    # F1_weighted = 0.0
    # p = 0
    # all_test_id = data_utility.get_all_unseen_types()
    # for i in all_test_id:
    #     if i in unseen_label_ids:
    #         F1_weighted += (b[i][1]/count_sum) * test_type_f1s[p]
    #         p += 1
    # print 'pattern baseline type averaged macro = {}'.format(F1_weighted)
    # print 'pattern baseline type micro = {}'.format(F1[2][2])

    epoch_type_f1_info = data_utility.type_f1s()
    epoch_type_f1_info.add_test_auc(unseen_label_ids, batch_data[-1], print_predict_y)

    #
    # print epoch_type_f1_info.test_auc_dict
    auc_array = []
    for k in epoch_type_f1_info.test_auc_dict:
        auc_array.append(epoch_type_f1_info.test_auc_dict[k])

    print('auc ave = {}'.format(np.average(auc_array)))


def baseline_predict(feature_data):
    ret = np.zeros((feature_data.shape[0], feature_data.shape[1]))

    max_num = 0

    for i in range(0, feature_data.shape[0]):
        for j in range(0, feature_data.shape[1]):
            # if feature_data[i][j][0] != 0.0:
            #     ret[i][j] = 1.0
            ret[i][j] = feature_data[i][j][2]
            if ret[i][j] > max_num:
                max_num = ret[i][j]

    ret[i][j] /= max_num

    return ret


if __name__ == "__main__":
    # emb_baseline()
    # pattern_baseline()
    # pattern_baseline_umls()
    emb_baseline_umls()
