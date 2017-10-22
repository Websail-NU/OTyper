import numpy as np

def gen_CV():
    test_ids = []
    with open('test_data_types.txt', 'r') as f:
        for line in f:
            test_ids.append(int(line.replace('\n','')))

    np.random.shuffle(test_ids)

    test_p = 0

    with open('CV_output.txt', 'w') as f:
        for i in range(0, 10):
            if len(test_ids) - test_p >= 8:
                c_test_ids = test_ids[test_p:test_p+4]
                test_p += 4
            else:
                c_test_ids = test_ids[test_p:test_p+5]
                test_p += 5

            c_rest_ids = []
            for j in range(0, 113):
                if not j in c_test_ids:
                    c_rest_ids.append(j)

            np.random.shuffle(c_rest_ids)

            c_dev_ids = c_rest_ids[0:10]

            c_train_ids = c_rest_ids[10:]

            f.write(' '.join(str(e) for e in c_train_ids))
            f.write('\n')
            f.write(' '.join(str(e) for e in c_dev_ids))
            f.write('\n')
            f.write(' '.join(str(e) for e in c_test_ids))
            f.write('\n')


if __name__ == "__main__":
    gen_CV()
