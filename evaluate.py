import numpy as np

def acc_hook(scores, y_data, epoch, train_flag, log_path):
    true_and_prediction = get_true_and_prediction(scores, y_data)
    if train_flag == 1:
        pre_str = 'train'
    elif train_flag == 0:
        pre_str = 'test'
    elif train_flag == 2:
        pre_str = 'dev'

    with open(log_path, 'a') as f:
        f.write('{} epoch = {}\n'.format(pre_str, epoch))
        # l1 = strict(true_and_prediction)
        l2 = loose_macro(true_and_prediction)
        l3 = loose_micro(true_and_prediction)
        # f.write("{}      strict (p,r,f1): {}  {}  {}\n".format(pre_str, *l1))
        f.write("{} loose macro (p,r,f1): {}  {}  {}\n".format(pre_str, *l2))
        f.write("{} loose micro (p,r,f1): {}  {}  {}\n".format(pre_str, *l3))


    ret_array = np.zeros((3,3))
    for i in range(0,3):
        ret_array[0][i] = 0.0
    for i in range(0,3):
        ret_array[1][i] = l2[i]
    for i in range(0,3):
        ret_array[2][i] = l3[i]

    return ret_array


def acc_hook_f1_break_down(scores, y_data, epoch, train_flag, log_path, prior=None):
    true_and_prediction, modified_scores = get_true_and_prediction(scores, y_data, prior=None)
    if train_flag == 1:
        pre_str = 'train'
    elif train_flag == 0:
        pre_str = 'test'
    elif train_flag == 2:
        pre_str = 'dev'

    with open(log_path, 'a') as f:
        f.write('{} epoch = {}\n'.format(pre_str, epoch))
        # l1 = strict(true_and_prediction)
        l2 = loose_macro(true_and_prediction)
        l3 = loose_micro(true_and_prediction)
        # f.write("{}      strict (p,r,f1): {}  {}  {}\n".format(pre_str, *l1))
        f.write("{} loose macro (p,r,f1): {}  {}  {}\n".format(pre_str, *l2))
        f.write("{} loose micro (p,r,f1): {}  {}  {}\n".format(pre_str, *l3))


    ret_array = np.zeros((3,3))
    for i in range(0,3):
        ret_array[0][i] = 0.0
    for i in range(0,3):
        ret_array[1][i] = l2[i]
    for i in range(0,3):
        ret_array[2][i] = l3[i]


    true_and_prediction_trans, _ = get_true_and_prediction(np.transpose(scores), np.transpose(y_data), prior)

    type_f1s = loose_macro_break_down(true_and_prediction_trans)[-1]

    l3 = loose_micro(true_and_prediction_trans)

    F1_num = l3[-1]

    return ret_array, type_f1s, F1_num


def trans_acc_hook(scores, y_data, epoch, train_flag):
    scores = np.transpose(scores)
    y_data = np.transpose(y_data)
    true_and_prediction = get_true_and_prediction(scores, y_data)
    if train_flag == 1:
        pre_str = 'train'
    elif train_flag == 0:
        pre_str = 'test'

    pl, rl, f1l = loose_macro_break_down(true_and_prediction)
    for i in range(0, len(pl)):
        print('i = {} prec = {}, recall = {}, F1 = {}'.format(i, pl[i], rl[i], f1l[i]))



def get_true_and_prediction(scores, y_data, prior, max_flag=False):
    if max_flag == False:
        predict_modify = np.zeros(scores.shape)
        true_and_prediction = []
        p = 0
        count = 0
        for score,true_label in zip(scores,y_data):
            predicted_tag = []
            true_tag = []
            for label_id,label_score in enumerate(list(true_label)):
                if label_score > 0:
                    true_tag.append(label_id)
            for label_id,label_score in enumerate(list(score)):
                if label_score >= 0.5:
                    predicted_tag.append(label_id)
            if not prior is None:
                selected_len = int(round(len(list(score))*prior[count]))

                sort_ids = np.argsort(list(score))[::-1]
                for i in range(0, selected_len):
                    if not sort_ids[i] in predicted_tag:
                        predicted_tag.append(int(sort_ids[i]))

            t_a = np.zeros(scores.shape[1])
            for e in predicted_tag:
                t_a[e] = 1.0
            predict_modify[p] = t_a
            p += 1
            true_and_prediction.append((true_tag, predicted_tag))
            count += 1
    else:
        pass
        # predict_modify = np.zeros(scores.shape)
        # true_and_prediction = []
        # p = 0
        # for score,true_label in zip(scores,y_data):
        #     predicted_tag = []
        #     true_tag = []
        #     for label_id,label_score in enumerate(list(true_label)):
        #         if label_score > 0:
        #             true_tag.append(label_id)
        #     lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        #     predicted_tag.append(lid)
        #     for label_id,label_score in enumerate(list(score)):
        #         if label_score >= 0.5:
        #             if label_id != lid:
        #                 predicted_tag.append(label_id)
        #     t_a = np.zeros(scores.shape[1])
        #     for e in predicted_tag:
        #         t_a[e] = 1.0
        #     predict_modify[p] = t_a
        #     p += 1
        #     true_and_prediction.append((true_tag, predicted_tag))



    return true_and_prediction, predict_modify
    # return true_and_prediction

def f1(p,r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )

def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    precision = recall = correct_num / num_entities
    return precision, recall, f1( precision, recall)

def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)

def loose_macro_break_down(true_and_prediction):
    ret_pl = []
    ret_rl = []
    ret_f1l = []

    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            ret_pl.append(len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels)))
        else:
            if len(true_labels):
                ret_pl.append(1.0)
            else:
                ret_pl.append(0.0)

        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            ret_rl.append(len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels)))
        else:
            r += 1.0
            ret_rl.append(1.0)

        ret_f1l.append(f1(ret_pl[-1], ret_rl[-1]))

    # if len(ret_f1l) < 20:
    #     for i in range(0, len(ret_f1l)):
    #         print '{}, {}, {}'.format(ret_pl[i], ret_rl[i], f1(ret_pl[i], ret_rl[i]))
    precision = p / num_entities
    recall = r / num_entities
    # return ret_pl, ret_rl, ret_f1l
    return precision, recall, f1(precision, recall), ret_f1l


def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    if num_predicted_labels != 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        if num_correct_labels == 0:
            precision = 1.0
        else:
            precision = 0.0
    if num_true_labels != 0:
        recall = num_correct_labels / num_true_labels
    else:
        recall = 1.0
    ret_num = np.zeros(3)
    ret_num[0] = num_correct_labels
    ret_num[1] = num_predicted_labels
    ret_num[2] = num_true_labels
    return precision, recall, f1(precision, recall), ret_num
