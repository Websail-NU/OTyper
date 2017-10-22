

def w2v_type_L1_norm():
    x = tf.placeholder(tf.float32, [None, 600], name = 'x')
    x1 = tf.slice(x,[0,0],[-1,300])
    x2 = tf.slice(x,[0,300],[-1,-1])
    x1_norm = tf.reshape(tf.norm(x1, ord = 1, axis = 1), [-1, 1])
    x2_norm = tf.reshape(tf.norm(x2, ord = 1, axis = 1), [-1, 1])
    x_with_norm = tf.concat([x, x1_norm, x2_norm], 1)

    dense_1 = tf.layers.dense(inputs=x_with_norm, units=150, activation=tf.nn.tanh, name = 'dense_1')

    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')
    W = tf.Variable(tf.tf.layers.dense([150, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(dense_1, W), b, name = 'y`_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    # loss = tf.reduce_mean((y - y_) * (y - y_), name = 'loss')
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w))

    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()

    figer = data_utility.figer_data()

    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 40*figer.total_batch_num):
            batch_xs, batch_ys, batch_ws = figer.next_shuffle_train_batch()
            _, print_loss = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})

            with open('loss_record.txt', 'a') as out_file:
                out_file.write('train: epoch= {}  loss= {}\n' \
                        .format(i, print_loss))
                # print ('train: epoch= {}  loss= {}\n' \
                #         .format(i, print_loss))


            if i % figer.total_batch_num == 10:
                get_validation_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_train_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_test_data_info(sess, figer, i, loss, x, y, y_, example_w)
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)


def w2v_type_diff():
    x = tf.placeholder(tf.float32, [None, 600], name = 'x')
    x1 = tf.slice(x,[0,0],[-1,300])
    x2 = tf.slice(x,[0,300],[-1,-1])

    x_diff = x1 - x2
    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')
    W = tf.Variable(tf.random_normal([300, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(x_diff, W), b, name = 'y_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    # loss = tf.reduce_mean((y - y_) * (y - y_), name = 'loss')
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w))

    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()

    figer = data_utility.figer_data()

    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 40*figer.total_batch_num):
            batch_xs, batch_ys, batch_ws = figer.next_shuffle_train_batch()
            _, print_loss = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})

            with open('loss_record.txt', 'a') as out_file:
                out_file.write('train: epoch= {}  loss= {}\n' \
                        .format(i, print_loss))
                # print ('train: epoch= {}  loss= {}\n' \
                #         .format(i, print_loss))

            if i % figer.total_batch_num == 10:
                get_validation_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_train_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_test_data_info(sess, figer, i, loss, x, y, y_, example_w)
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)



def w2v_type_linear():
    x = tf.placeholder(tf.float32, [None, 600], name = 'x')
    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')
    W = tf.Variable(tf.random_normal([600, 1]), name = 'W')
    # W = tf.Variable(tf.zeros([600, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(x, W), b, name = 'y_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w))

    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')

    figer = data_utility.figer_data()

    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 40*figer.total_batch_num):
            # batch_xs, batch_ys, batch_ws = figer.next_shuffle_train_batch_random_entity()

            batch_xs, batch_ys, batch_ws = figer.next_shuffle_train_batch()
            _, print_loss = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})

            with open('loss_record.txt', 'a') as out_file:
                out_file.write('train: epoch= {}  loss= {}\n' \
                        .format(i, print_loss))

            if i % figer.total_batch_num == 10:
                get_validation_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_train_data_info(sess, figer, i, loss, x, y, y_, example_w)
                get_test_data_info(sess, figer, i, loss, x, y, y_, example_w)
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)

def get_train_data_info(sess, figer, epoch, loss, x, y, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.test_train_shuffle_pos = 0
    predict_count = np.zeros([113,2])
    raw_count = np.zeros([113,2])
    for i in range(0, int(figer.total_batch_num * 0.8)):
        batch_xs, batch_ys, batch_ws, type_ids = figer.next_shuffle_test_train_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('train : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.8), print_loss))
        for j in range(0, len(print_y)):
            if print_y[j] > 0.5:
                predict_count[type_ids[j]][1] += 1
            else:
                predict_count[type_ids[j]][0] += 1
            if batch_ys[j] > 0.5:
                raw_count[type_ids[j]][1] += 1
            else:
                raw_count[type_ids[j]][0] += 1
            if batch_ys[j][0] == 0.0:
                zero_R_count += 1.0
            if batch_ys[j][0] == 1.0:
                if print_y[j] >=0.5:
                    TP += 1.0
                else:
                    FN += 1.0
            else:
                if print_y[j] >=0.5:
                    FP += 1.0
                else:
                    TN += 1.0

    if TP + FN != 0:
        NN_rec = float(TP) / float(TP + FN)
    else:
        NN_rec = 1.0
    if TP + FP != 0:
        NN_prec = float(TP) / float(TP + FP)
    else:
        NN_prec = 1.0

    if NN_prec + NN_rec == 0.0:
        NN_F1 = 0.0
    else:
        NN_F1 = 2.0 * float(NN_prec * NN_rec) / float(NN_prec + NN_rec)
    zero_R_prec = float(zero_R_count)/ float(TP + FN + FP + TN)
    ave_loss = total_loss/float(figer.total_batch_num * 0.8)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))
    with open('prior.txt', 'a') as out_file:
        out_file.write('train: epoch= {}\n'.format(epoch))
        for i in range(0, 113):
            out_file.write('train_predict_type {}: N_cnt: {} Y_cnt: {} raw: N_cnt: {} Y_cnt: {}\n' \
                .format(i, predict_count[i][0], predict_count[i][1], raw_count[i][0], raw_count[i][1]))


def get_validation_data_info(sess, figer, epoch, loss, x, y, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    m_TP = 0.0
    m_FN = 0.0
    m_FP = 0.0
    m_TN = 0.0
    total_loss = 0.0
    figer.validation_shuffle_pos = 0
    predict_count = np.zeros([113,2])
    m_predict_count = np.zeros([113,2])
    raw_count = np.zeros([113,2])
    Y_prior = []
    with open('Y_prior.txt', 'r') as out_file:
        lines = out_file.readlines()
        for e in lines:
            Y_prior.append(float(e))

    for i in range(0, int(figer.total_batch_num * 0.2)):
        batch_xs, batch_ys, batch_ws, type_ids = figer.next_shuffle_validation_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss

        with open('temp.txt', 'w') as out_file:
            out_file.write('validation : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.2), print_loss))


        for j in range(0, len(print_y)):
            if print_y[j] > 0.5:
                predict_count[type_ids[j]][1] += 1
            else:
                predict_count[type_ids[j]][0] += 1
            if batch_ys[j] > 0.5:
                raw_count[type_ids[j]][1] += 1
            else:
                raw_count[type_ids[j]][0] += 1
            if batch_ys[j][0] == 0.0:
                zero_R_count += 1.0
            if batch_ys[j][0] == 1.0:
                if print_y[j] >=0.5:
                    TP += 1.0
                else:
                    FN += 1.0
            else:
                if print_y[j] >=0.5:
                    FP += 1.0
                else:
                    TN += 1.0

#       top prior select
        node_dict = {}
        for j in range(0, len(print_y)):
            node = [j, print_y[j]]
            if not type_ids[j] in node_dict:
                node_dict[type_ids[j]] = []
            node_dict[type_ids[j]].append(node)

#       hack! change the meaning of print_y
        for key in node_dict:
            type_Y_prior = Y_prior[key]
            top_length = round(type_Y_prior * len(node_dict[key]))
            sorted_list = sorted(node_dict[key], key=itemgetter(1), reverse=True)
            for k in range(0, len(node_dict[key])):
                if k < top_length:
                    print_y[sorted_list[k][0]] = 1.0
                else:
                    print_y[sorted_list[k][0]] = 0.0

        for j in range(0, len(print_y)):
            if print_y[j] > 0.5:
                m_predict_count[type_ids[j]][1] += 1
            else:
                m_predict_count[type_ids[j]][0] += 1
            if batch_ys[j][0] == 1.0:
                if print_y[j] >=0.5:
                    m_TP += 1.0
                else:
                    m_FN += 1.0
            else:
                if print_y[j] >=0.5:
                    m_FP += 1.0
                else:
                    m_TN += 1.0

    if TP + FN != 0:
        NN_rec = float(TP) / float(TP + FN)
    else:
        NN_rec = 1.0

    if TP + FP != 0:
        NN_prec = float(TP) / float(TP + FP)
    else:
        NN_prec = 1.0

    if NN_prec + NN_rec == 0.0:
        NN_F1 = 0.0
    else:
        NN_F1 = 2.0 * float(NN_prec * NN_rec) / float(NN_prec + NN_rec)

#   F1 for modified validation error

    if m_TP + m_FN != 0.0:
        m_NN_rec = float(m_TP) / float(m_TP + m_FN)
    else:
        m_NN_rec = 1.0

    if m_TP + m_FP != 0.0:
        m_NN_prec = float(m_TP) / float(m_TP + m_FP)
    else:
        m_NN_prec = 1.0

    if m_NN_prec + m_NN_rec == 0.0:
        m_NN_F1 = 0.0
    else:
        m_NN_F1 = 2.0 * float(m_NN_prec * m_NN_rec) / float(m_NN_prec + m_NN_rec)


    zero_R_prec = float(zero_R_count)/ float(TP + FN + FP + TN)
    ave_loss = total_loss/float(figer.total_batch_num * 0.2)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('validation: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}  ' \
                        'm_NN_prec= {}  m_NN_rec= {}  m_NN_F1= {}  m_TP= {}  m_FN= {}  m_FP= {}  m_TN= {} \n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN, m_NN_prec, m_NN_rec, m_NN_F1, m_TP, m_FN, m_FP, m_TN))

    with open('prior.txt', 'a') as out_file:
        out_file.write('validation: epoch= {}\n'.format(epoch))
        for i in range(0, 113):
            out_file.write('validation_predict_type {}: N_cnt: {} Y_cnt: {} m_N_cnt: {} m_Y_cnt: {} raw: N_cnt: {} Y_cnt: {}\n' \
                .format(i, predict_count[i][0], predict_count[i][1], m_predict_count[i][0], m_predict_count[i][1], raw_count[i][0], raw_count[i][1]))


def get_test_data_info(sess, figer, epoch, loss, x, y, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    m_TP = 0.0
    m_FN = 0.0
    m_FP = 0.0
    m_TN = 0.0
    total_loss = 0.0
    figer.test_order_pos = 0
    predict_count = np.zeros([113,2])
    m_predict_count = np.zeros([113,2])
    raw_count = np.zeros([113,2])
    Y_prior = []
    with open('Y_prior.txt', 'r') as out_file:
        lines = out_file.readlines()
        for e in lines:
            Y_prior.append(float(e))
    for i in range(0, figer.total_batch_num):
        batch_xs, batch_ys, batch_ws, type_ids = figer.next_order_test_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('test : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num), print_loss))
        for j in range(0, len(print_y)):
            if print_y[j] > 0.5:
                predict_count[type_ids[j]][1] += 1
            else:
                predict_count[type_ids[j]][0] += 1
            if batch_ys[j] > 0.5:
                raw_count[type_ids[j]][1] += 1
            else:
                raw_count[type_ids[j]][0] += 1
            if batch_ys[j][0] == 0.0:
                zero_R_count += 1.0
            if batch_ys[j][0] == 1.0:
                if print_y[j] >=0.5:
                    TP += 1.0
                else:
                    FN += 1.0
            else:
                if print_y[j] >=0.5:
                    FP += 1.0
                else:
                    TN += 1.0


#       top prior select
        node_dict = {}
        for j in range(0, len(print_y)):
            node = [j, print_y[j]]
            if not type_ids[j] in node_dict:
                node_dict[type_ids[j]] = []
            node_dict[type_ids[j]].append(node)

#       hack! change the meaning of print_y
        for key in node_dict:
            type_Y_prior = Y_prior[key]
            top_length = round(0.03 * len(node_dict[key]))
            sorted_list = sorted(node_dict[key], key=itemgetter(1), reverse=True)
            for k in range(0, len(node_dict[key])):
                if k < top_length:
                    print_y[sorted_list[k][0]] = 1.0
                else:
                    print_y[sorted_list[k][0]] = 0.0

        for j in range(0, len(print_y)):
            if print_y[j] > 0.5:
                m_predict_count[type_ids[j]][1] += 1
            else:
                m_predict_count[type_ids[j]][0] += 1
            if batch_ys[j][0] == 1.0:
                if print_y[j] >=0.5:
                    m_TP += 1.0
                else:
                    m_FN += 1.0
            else:
                if print_y[j] >=0.5:
                    m_FP += 1.0
                else:
                    m_TN += 1.0


    if TP + FN != 0:
        NN_rec = float(TP) / float(TP + FN)
    else:
        NN_rec = 1.0
    if TP + FP != 0:
        NN_prec = float(TP) / float(TP + FP)
    else:
        NN_prec = 1.0

    if NN_prec + NN_rec == 0.0:
        NN_F1 = 0.0
    else:
        NN_F1 = 2.0 * float(NN_prec * NN_rec) / float(NN_prec + NN_rec)

#   F1 for modified validation error

    if m_TP + m_FN != 0.0:
        m_NN_rec = float(m_TP) / float(m_TP + m_FN)
    else:
        m_NN_rec = 1.0

    if m_TP + m_FP != 0.0:
        m_NN_prec = float(m_TP) / float(m_TP + m_FP)
    else:
        m_NN_prec = 1.0

    if m_NN_prec + m_NN_rec == 0.0:
        m_NN_F1 = 0.0
    else:
        m_NN_F1 = 2.0 * float(m_NN_prec * m_NN_rec) / float(m_NN_prec + m_NN_rec)

    zero_R_prec = float(zero_R_count)/ float(TP + FN + FP + TN)
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}  ' \
                        'm_NN_prec= {}  m_NN_rec= {}  m_NN_F1= {}  m_TP= {}  m_FN= {}  m_FP= {}  m_TN= {} \n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN, m_NN_prec, m_NN_rec, m_NN_F1, m_TP, m_FN, m_FP, m_TN))


    with open('prior.txt', 'a') as out_file:
        out_file.write('test: epoch= {}\n'.format(epoch))
        for i in range(0, 113):
            out_file.write('test_predict_type {}: N_cnt: {} Y_cnt: {} m_N_cnt: {} m_Y_cnt: {} raw: N_cnt: {} Y_cnt: {}\n' \
                .format(i, predict_count[i][0], predict_count[i][1], m_predict_count[i][0], m_predict_count[i][1], raw_count[i][0], raw_count[i][1]))



#   fully connected

def w2v_type_fully():
    x = tf.placeholder(tf.float32, [None, 600], name = 'x')
    dense_1 = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.relu, name = 'dense_1')
    dense_2 = tf.layers.dense(inputs=dense_1, units=150, activation=tf.nn.tanh, name = 'dense_2')
    W = tf.Variable(tf.zeros([150, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y = tf.sigmoid(tf.matmul(dense_2, W) + b, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    loss = tf.reduce_mean((y - y_) * (y - y_), name = 'loss')
    train_step = tf.train.AdamOptimizer(0.003).minimize(loss, name = 'train_step')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    figer = data_utility.figer_data()
    test_xs, test_ys = figer.get_test_data(0, 10000)

    saver = tf.train.Saver()
    for i in range(0,10000):
        batch_xs, batch_ys = figer.next_batch()
        _, print_loss, print_y = sess.run([train_step, loss, y], feed_dict={x: batch_xs, y_: batch_ys})

        zero_R_count = 0.0
        NN_count = 0.0
        for j in range(0, len(print_y)):
            if batch_ys[j][0] == 0.0:
                zero_R_count += 1.0
            if abs(print_y[j] - batch_ys[j][0]) < 0.5:
                NN_count += 1.0

        zero_R_acc = zero_R_count / float(len(print_y))
        NN_acc = NN_count / float(len(print_y))

        with open('loss_record.txt', 'a') as out_file:
            out_file.write('epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                            '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')
        print ('epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                        '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')

        print_loss, print_y = sess.run([loss, y], feed_dict={x: test_xs, y_: test_ys})
        zero_R_count = 0.0
        NN_count = 0.0
        for j in range(0, len(print_y)):
            if test_ys[j][0] == 0.0:
                zero_R_count += 1.0
            if abs(print_y[j] - test_ys[j][0]) < 0.5:
                NN_count += 1.0

        zero_R_acc = zero_R_count / float(len(print_y))
        NN_acc = NN_count / float(len(print_y))

        if i%10 == 0:
            with open('loss_record.txt', 'a') as out_file:
                out_file.write('test: epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                                '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')

        if i % 100 == 0:
            model_name = 'model/model_' + str(i) + '.ckpt'
            saver.save(sess, model_name)

def w2v_type_restore():
    sess=tf.Session()
    saver = tf.train.import_meta_graph('model/model_56000.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))
    W = sess.run('W:0')

    total = 0.0
    for i in range(0,300):
        total += abs(W[i])

    print total/float(300)

    total = 0.0
    for i in range(0,300):
        total += abs(W[i+300])

    print total/float(300)


# gen cached data

def read_csv(file_path, all_entity_str, all_type_str):
    entity_list = []

    with open(file_path, 'r') as raw_file:
        with codecs.open(file_path, 'rU') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
            csv.field_size_limit(sys.maxsize)
            next(reader)
            next(raw_file)
            for row in reader:
                v = entity(row, next(raw_file))
                if v.class_string in all_type_str and v.entity_string in all_entity_str:
                    entity_list.append(v)

    return entity_list


def read_all_csv(dir_path):
    all_entity_str = set()

    with open('data/all_entity_str.txt', 'r') as f:
        for line in f:
            all_entity_str.add(line.replace('\n',''))

    all_type_str = set()
    with open('data/all_type_str.txt', 'r') as f:
        for line in f:
            all_type_str.add(line.replace('\n',''))

    all_entity_list = []
    count = 0
    with open('data/cached_webisa.txt', 'w') as f:
        f.write('_id,instance,class,frequency,pidspread,pldspread,modifications\n')
    for file_name in os.listdir(dir_path):
        entity_list = read_csv(os.path.join(dir_path, file_name), all_entity_str, all_type_str)
        # entity_list = read_csv('data/cached_webisa.txt', all_entity_str, all_type_str)
        with open('data/cached_webisa.txt', 'a') as f:
            for e in entity_list:
                f.write(e.raw_row)
        # all_entity_list += entity_list
        count += 1
        if count % 10 == 0:
            print count

    return all_entity_list



def gen_all_entity_str():
    ret_set = set()

    with open('data/state_of_the_art_dev_word_with_context.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            for word in line.split():
                if word.isalpha():
                    ret_set.add(word)

    with open('data/state_of_the_art_test_word_with_context.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            for word in line.split():
                if word.isalpha():
                    ret_set.add(word)

    with open('data/state_of_the_art_train_word_with_context.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            for word in line.split():
                if word.isalpha():
                    ret_set.add(word)

    with open('./data/all_entity_str.txt', 'w') as f:
        for word in ret_set:
            f.write('{}\n'.format(word))


# deleted code in create_model()
    # if entity_type_feature_flag == 0 and exact_entity_type_feature_flag == 0:
    #     final_logit = tf.matmul(input_representation, label_id2emb_matrix_dropout, transpose_b = True, name='final_logit')
    # elif entity_type_feature_flag == 1 and exact_entity_type_feature_flag == 0:
    #     logit = tf.matmul(input_representation, label_id2emb_matrix_dropout, transpose_b = True, name='logit')
    #     logit_expand = tf.expand_dims(logit, axis=-1)
    #     concate_logit = tf.concat([logit_expand, entity_type_features], -1)
    #     final_W = tf.get_variable(initializer=tf.random_uniform([4, 1], minval=-0.01, maxval=0.01), name='final_W')
    #     final_logit = tf.squeeze(tf.layers.dense(concate_logit, 1, name='final_logit'), axis=-1)
    # elif entity_type_feature_flag == 0 and exact_entity_type_feature_flag == 1:
    #     logit = tf.matmul(input_representation, label_id2emb_matrix_dropout, transpose_b = True, name='logit')
    #     logit_expand = tf.expand_dims(logit, axis=-1)
    #     concate_logit = tf.concat([logit_expand, exact_entity_type_features], -1)
    #     final_W = tf.get_variable(initializer=tf.random_uniform([4, 1], minval=-0.01, maxval=0.01), name='final_W')
    #     final_logit = tf.squeeze(tf.layers.dense(concate_logit, 1, name='final_logit'), axis=-1)
    # elif entity_type_feature_flag == 1 and exact_entity_type_feature_flag == 1:
    #     logit = tf.matmul(input_representation, label_id2emb_matrix_dropout, transpose_b = True, name='logit')
    #     logit_expand = tf.expand_dims(logit, axis=-1)
    #     concate_logit = tf.concat([logit_expand, entity_type_features, exact_entity_type_features], -1)
    #     final_W = tf.get_variable(initializer=tf.random_uniform([7, 1], minval=-0.01, maxval=0.01), name='final_W')
    #     final_logit = tf.squeeze(tf.layers.dense(concate_logit, 1, name='final_logit'), axis=-1)


def gen_all_gid():
    with open('umls_data/all_entity_gid_list.pkl', 'r') as f:
        all_entity_gid_list = pickle.load(f)
    s = set()
    for e in all_entity_gid_list:
        s.update(e)

    ret = []
    for e in s:
        ret.append(e)
    with open('umls_data/all_entity_gid_list_new.pkl', 'w') as f:
        pickle.dump(ret, f)
    print len(ret)


    with open('umls_data/all_entity_name_list.pkl', 'r') as f:
        all_entity_name_list = pickle.load(f)
    s = set()
    for e in all_entity_name_list:
        s.update(e)

    ret = []
    for e in s:
        ret.append(e)
    with open('umls_data/all_entity_name_list_new.pkl', 'w') as f:
        pickle.dump(ret, f)

    print len(ret)
