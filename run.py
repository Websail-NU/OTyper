import os

def run_helper(model_name, feature_flag, exact_entity_type_feature_flag, type_only_feature_flag, list_range, cmd_flag):
    entity_type_feature_flag = 0

    for id_select_flag in list_range:
        if cmd_flag == 'openner':
            cmd = 'python seen_type_dot_distance_label_matrix.py ' + \
                    str(model_name) + ' ' + str(feature_flag) + ' ' + \
                    str(entity_type_feature_flag) + ' ' + str(exact_entity_type_feature_flag) + \
                    ' ' + str(type_only_feature_flag) + ' ' + str(id_select_flag) + \
                    ' -auto_gen_log_path 1'
        elif cmd_flag == 'base_line':
            cmd = 'python baseline.py ' + \
                    str(model_name) + ' ' + str(feature_flag) + ' ' + \
                    str(entity_type_feature_flag) + ' ' + str(exact_entity_type_feature_flag) + \
                    ' ' + str(type_only_feature_flag) + ' ' + str(id_select_flag) + \
                    ' -auto_gen_log_path 1'

        print '{}'.format(cmd)
        os.system(cmd)

def run():
    # run_helper('emb_sub', 0, 0, 0, range(20, 30), 'base_line')
    # run_helper('attention', 1, 1, 1, range(20, 30), 'openner')
    # run_helper('attention', 0, 1, 1, range(20, 30), 'openner')
    # run_helper('attention', 1, 0, 1, range(20, 30), 'openner')
    # run_helper('attention', 1, 1, 0, range(20, 30), 'openner')
    # run_helper('attention', 1, 1, 1, range(30, 30+11), 'openner')
    # run_helper('attention', 1, 1, 1, range(50, 50+11), 'openner')

    run_helper('attention', 0, 0, 0, range(3, 5), 'openner')
    run_helper('attention', 0, 0, 1, range(3, 5), 'openner')
    run_helper('attention', 0, 1, 0, range(3, 5), 'openner')

    run_helper('attention', 0, 0, 0, range(5, 7), 'openner')
    run_helper('attention', 0, 0, 1, range(5, 7), 'openner')
    run_helper('attention', 0, 1, 0, range(5, 7), 'openner')


if __name__ == "__main__":
    run()
