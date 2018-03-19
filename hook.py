# -*- coding: utf-8 -*-
import sys

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
    return precision, recall, f1( precision, recall)

def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)

def get_true_and_prediction(scores, y_data):
    true_and_prediction = []
    for score,true_label in zip(scores,y_data):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id,label_score in enumerate(list(score)):
            if label_score > 0.5:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))
    return true_and_prediction

def acc_hook(scores, y_data):
    true_and_prediction = get_true_and_prediction(scores, y_data)
    with open('test_F1_results.txt', 'a') as wf:
        print("     strict (p,r,f1):",strict(true_and_prediction))
        print("loose macro (p,r,f1):",loose_macro(true_and_prediction))
        print("loose micro (p,r,f1):",loose_micro(true_and_prediction))
        wf.write("     strict (p,r,f1):{}\n".format(strict(true_and_prediction)))
        wf.write("loose macro (p,r,f1):{}\n".format(loose_macro(true_and_prediction)))
        wf.write("loose micro (p,r,f1):{}\n".format(loose_micro(true_and_prediction)))
