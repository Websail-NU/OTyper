import cPickle as pickle

def get_single_log_file_results(data_id):
    folder = './umls_type_f1_file/'
    log_folder = './umls_log_files/'
    model_string = 'ave_'
    flag_string = '0_0_0_0_'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    final_test_dict = {}
    data_id = str(data_id)
        # get_one_CV_results(folder, model_string, flag_string, cv, dicts)
    test_dict, F1_num, final_auc_dict = get_one_CV_results_test_F1s(folder, log_folder, model_string, flag_string, data_id)
    final_test_dict.update(test_dict)

    with open('data/test_type_count.pkl', 'r') as f:
        b = pickle.load(f)

    count_sum = 0
    for k in final_test_dict:
        count_sum += b[int(k)][1]
    #
    # F1_weighted = 0.0
    # for k in final_test_dict:
    #     F1_weighted += (b[int(k)][1]/count_sum) * final_test_dict[k]
    #
    # print 'OpenNER single file type average Macro {}'.format(F1_weighted)


    auc_array = []
    for  k in final_auc_dict:
        key = k
        auc_array.append((b[int(k)][1]/count_sum) * final_auc_dict[k])

    print 'average auc = {} {:.4f}'.format(key, np.sum(auc_array))

    return final_auc_dict

    # print '---popularity---'
    # print 'id---name---freq---auc---'
    # dicts = joblib.load('/home/zys133/knowledge_base/NFGEC/data/Wiki/dicts_figer.pkl')
    # for k in final_auc_dict:
    #     print '{}\t{}\t{}\t{:.4f}'.format(k, dicts['id2label'][k], b[k][1], final_auc_dict[k])




def get_one_CV_results_test_F1s(folder_pkl, folder_log, model_string, flag_string, cv):
    file_path = folder_pkl + model_string + flag_string + cv + '.pickle'
    with open(file_path, 'r') as data_file:
        data = pickle.load(data_file)
    file_path = folder_log + model_string + flag_string + cv + '.txt'
    best_id = get_best_epoch(file_path)
    return data[best_id].test_f1_dict, data[best_id].test_F1_num, data[best_id].test_auc_dict
    # return data[best_id].test_f1_dict, np.zeros(3)


def get_best_epoch(file_path):
    count = 0
    best_f1 = -0.1
    ret_id = -1
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.replace('\n','')
            if count % 9 == 5:
                t_f1 = line.split(' ')[-1]
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    ret_id = int(count / 9)
            count += 1

    return ret_id

if __name__ == "__main__":
    read
