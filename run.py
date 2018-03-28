import argparse
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

        print('{}'.format(cmd))
        os.system(cmd)


def run_helper_MSH(model_name, feature_flag, exact_entity_type_feature_flag, type_only_feature_flag, list_range, cmd_flag):
    entity_type_feature_flag = 0

    for id_select_flag in list_range:
        if cmd_flag == 'openner':
            cmd = 'python umls_seen_type_dot_distance_label_matrix.py ' + \
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

        print('{}'.format(cmd))
        os.system(cmd)

def run():
    # run_helper('emb_sub', 0, 0, 0, range(20, 30), 'base_line')
    # run_helper('attention', 1, 1, 1, range(20, 30), 'openner')
    # run_helper('attention', 0, 1, 1, range(20, 30), 'openner')
    # run_helper('attention', 1, 0, 1, range(20, 30), 'openner')
    # run_helper('attention', 1, 1, 0, range(20, 30), 'openner')
    # run_helper('attention', 1, 1, 1, range(30, 30+11), 'openner')
    # run_helper('attention', 1, 1, 1, range(50, 50+11), 'openner')

    # run_helper('attention', 0, 0, 0, range(3, 5), 'openner')
    # run_helper('attention', 0, 0, 1, range(3, 5), 'openner')
    # run_helper('attention', 0, 1, 0, range(3, 5), 'openner')
    #
    # run_helper('attention', 0, 0, 0, range(5, 7), 'openner')
    # run_helper('attention', 0, 0, 1, range(5, 7), 'openner')
    # run_helper('attention', 0, 1, 0, range(5, 7), 'openner')

#    run_helper('attention', 1, 1, 1, range(10, 20), 'openner')
    # run_helper('attention', 1, 1, 1, range(10, 20), 'openner')


    run_helper_MSH('attention', 0, 1, 0, range(10,20), 'openner')

    # run_helper_MSH('attention', 0, 1, 0, range(10,20), 'openner')
    # run_helper_MSH('emb_sub', 0, 0, 0, range(10, 20), 'base_line')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_flag', help='which set of data to train', choices=['FIGER','MSH'])
    parser.add_argument('mention_feature', help='use mention feature? 1 for on, 0 for off', type=int, choices=[0, 1])
    parser.add_argument('entity_type_feature', help='use entity type feature?, 1 for on, 0 for off', type=int, choices=[0, 1])
    parser.add_argument('type_only_feature', help='use type only feature?, 1 for on, 0 for off', type=int, choices=[0, 1])

    args = parser.parse_args()

    mention_feature = args.mention_feature
    entity_type_feature = args.entity_type_feature
    type_only_feature = args.type_only_feature

    if args.data_flag == 'FIGER':
        # run_helper('attention', 1, 1, 1, range(10, 20), 'openner')
        run_helper('attention', mention_feature, entity_type_feature, type_only_feature, range(10, 20), 'openner')
    elif args.data_flag == 'MSH':
        # run_helper_MSH('attention', 0, 1, 0, range(10, 20), 'openner')
        run_helper_MSH('attention', mention_feature, entity_type_feature, type_only_feature, range(10, 20), 'openner')
    else:
        print('unknown argument')
