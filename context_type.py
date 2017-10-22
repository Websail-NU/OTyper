import tensorflow as tf
import numpy as np
import data_utility

def context_type_CNN():
    context = tf.placeholder(tf.float32, [None, 50, 300], name = 'context')
    pos_emb = tf.placeholder(tf.float32, [None, 50, 50], name = 'lengths_emb')
    x = tf.concat([context, pos_emb], 2, name = 'x')
    conv1 = tf.layers.conv1d(inputs=x, filters=300, kernel_size=[3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[50], strides=1, name = 'pool1')
    pool_last = tf.reshape(pool1, [-1,300])
    type_emb = tf.placeholder(tf.float32, [None, 300], name = 'type_emb')
    concat = tf.concat([pool_last, type_emb], 1, name = 'concat')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')
    W = tf.Variable(tf.zeros([600, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(concat, W), b, name = 'y_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w), name = 'loss')
    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')

    figer = data_utility.figer_data()
    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 40*figer.total_batch_num):
            batch_cs, batch_pos, batch_ls, batch_ys, batch_ws = figer.next_shuffle_train_context_batch_CNN()
            _, print_loss = sess.run([train_step, loss], feed_dict={context: batch_cs, pos_emb: batch_pos, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
            with open('loss_record.txt', 'a') as out_file:
                out_file.write('train: epoch= {}  loss= {}\n' \
                        .format(i, print_loss))
            if i % figer.total_batch_num == 10:
                get_train_data_info_CNN(sess, figer, i, loss, y, context, pos_emb, type_emb, y_, example_w)
                get_validation_data_info_CNN(sess, figer, i, loss, y, context, pos_emb, type_emb, y_, example_w)
                get_test_data_info_CNN(sess, figer, i, loss, y, context, pos_emb, type_emb, y_, example_w)
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)


def context_type_RNN():

    context = tf.placeholder(tf.float32, [None, None, 300], name = 'context')
    cell = tf.contrib.rnn.BasicLSTMCell(300)
    batch_size = tf.shape(context)[1]
    initial_state = cell.zero_state(batch_size, tf.float32)
    sequence_length = tf.placeholder(tf.int32, [None], name = 'sequence_length')
    outputs, _ = tf.nn.dynamic_rnn(cell, context, sequence_length = sequence_length, initial_state=initial_state, time_major=True, dtype=tf.float32)
    final_outputs = select_rnn(outputs, sequence_length - 1)

    type_emb = tf.placeholder(tf.float32, [None, 300], name = 'type_emb')

    x = tf.concat([final_outputs, type_emb], 1, name = 'x')

    W = tf.Variable(tf.zeros([600, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(x, W), b, name = 'y_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')

    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w), name = 'loss')
#    train_step = tf.train.AdamOptimizer(0.003).minimize(loss, name = 'train_step')
    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')

    figer = data_utility.figer_data()
    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 40*figer.total_batch_num):
            batch_cs, batch_cs_lengths, batch_ls, batch_ys, batch_ws = figer.next_shuffle_train_context_batch()
            _, print_loss = sess.run([train_step, loss], feed_dict={context: batch_cs, sequence_length: batch_cs_lengths, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
            with open('loss_record.txt', 'a') as out_file:
                out_file.write('train: epoch= {}  loss= {}\n' \
                        .format(i, print_loss))

            if i % figer.total_batch_num == 10:
                get_train_data_info(sess, figer, i, loss, y, context, sequence_length, type_emb, y_, example_w)
                get_validation_data_info(sess, figer, i, loss, y, context, sequence_length, type_emb, y_, example_w)
                get_test_data_info(sess, figer, i, loss, y, context, sequence_length, type_emb, y_, example_w)
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)


def get_train_data_info(sess, figer, epoch, loss, y, context, sequence_length, type_emb, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.train_context_shuffle_pos = 0
    for i in range(0, int(figer.total_batch_num * 0.8)):
        if np.random.uniform() > 0.3:
            figer.train_context_shuffle_pos = (figer.train_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_cs_lengths, batch_ls, batch_ys, batch_ws = figer.next_shuffle_train_context_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, sequence_length: batch_cs_lengths, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('train : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.8), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)


def get_validation_data_info(sess, figer, epoch, loss, y, context, sequence_length, type_emb, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.validation_context_shuffle_pos = 0
    for i in range(0, int(figer.total_batch_num * 0.2)):
        if np.random.uniform() > 0.3:
            figer.validation_context_shuffle_pos = (figer.validation_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_cs_lengths, batch_ls, batch_ys, batch_ws = figer.next_shuffle_validation_context_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, sequence_length: batch_cs_lengths, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('validation : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.2), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('validation: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)

def get_test_data_info(sess, figer, epoch, loss, y, context, sequence_length, type_emb, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.test_context_shuffle_pos = 0
    for i in range(0, figer.total_batch_num):
        if np.random.uniform() > 0.3:
            figer.test_context_shuffle_pos = (figer.test_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_cs_lengths, batch_ls, batch_ys, batch_ws = figer.next_shuffle_test_context_batch()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, sequence_length: batch_cs_lengths, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('test : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)



def get_train_data_info_CNN(sess, figer, epoch, loss, y, context, pos_emb, type_emb, y_, example_w):
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.train_context_shuffle_pos = 0
    for i in range(0, int(figer.total_batch_num * 0.8)):
        if np.random.uniform() > 0.3:
            figer.train_context_shuffle_pos = (figer.train_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_pos, batch_ls, batch_ys, batch_ws = figer.next_shuffle_train_context_batch_CNN()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, pos_emb: batch_pos, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('train : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.8), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)


def get_validation_data_info_CNN(sess, figer, epoch, loss, y, context, pos_emb, type_emb, y_, example_w):
    pass
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.validation_context_shuffle_pos = 0
    for i in range(0, int(figer.total_batch_num * 0.2)):
        if np.random.uniform() > 0.3:
            figer.validation_context_shuffle_pos = (figer.validation_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_pos, batch_ls, batch_ys, batch_ws = figer.next_shuffle_validation_context_batch_CNN()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, pos_emb: batch_pos, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('validation : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num * 0.2), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('validation: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)

def get_test_data_info_CNN(sess, figer, epoch, loss, y, context, pos_emb, type_emb, y_, example_w):
    pass
    zero_R_count = 0.0
    NN_count = 0.0
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    total_loss = 0.0
    figer.test_context_shuffle_pos = 0
    for i in range(0, figer.total_batch_num):
        if np.random.uniform() > 0.3:
            figer.test_context_shuffle_pos = (figer.test_context_shuffle_pos + figer.batch_size * 113) % (len(figer.Words)*113)
            continue
        batch_cs, batch_pos, batch_ls, batch_ys, batch_ws = figer.next_shuffle_test_context_batch_CNN()
        print_loss, print_y = sess.run([loss, y], feed_dict={context: batch_cs, pos_emb: batch_pos, type_emb: batch_ls, y_: batch_ys, example_w: batch_ws})
        total_loss += print_loss
        with open('temp.txt', 'w') as out_file:
            out_file.write('test : epoch= {}/{}  loss= {}\n' \
                .format(i, int(figer.total_batch_num), print_loss))
        for j in range(0, len(print_y)):
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
    ave_loss = total_loss/float(figer.total_batch_num)

    with open('loss_record.txt', 'a') as out_file:
        out_file.write('test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

    # print 'test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
    #         .format(epoch, ave_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)



def select_rnn(tensor, time_step):
    """return tensor at the time_step (time major). This is similar to numpy
    tensor[time_step, :, :] where time_step can be 1D array."""
    time_step = tf.expand_dims(time_step, axis=-1)
    range_ = tf.expand_dims(tf.range(start=0, limit=tf.shape(tensor)[1]), axis=-1)
    idx = tf.concat([time_step, range_], axis=-1)
    return tf.gather_nd(tensor, idx)


def context_type():

    context = tf.placeholder(tf.float32, [15, None, 300], name = 'context')
    cell = tf.contrib.rnn.BasicLSTMCell(300)
    batch_size = tf.shape(context)[1]
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, context, initial_state=initial_state, time_major=True, dtype=tf.float32)
    final_outputs = outputs[-1]

    type_emb = tf.placeholder(tf.float32, [None, 300], name = 'type_emb')

    x = tf.concat([final_outputs, type_emb], 1, name = 'x')

    dense_1 = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.relu, name = 'dense_1')
    dense_2 = tf.layers.dense(inputs=dense_1, units=150, activation=tf.nn.tanh, name = 'dense_2')
    W = tf.Variable(tf.zeros([150, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y = tf.sigmoid(tf.matmul(dense_2, W) + b, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    loss = tf.reduce_mean((y - y_) * (y - y_), name = 'loss')
#    train_step = tf.train.AdamOptimizer(0.003).minimize(loss, name = 'train_step')
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, name = 'train_step')

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        figer = data_utility.figer_data()
        test_cs, test_ls, test_ys = figer.get_context_test_data(0, 10000)

        saver = tf.train.Saver()
        for i in range(0,10000):
            batch_cs, batch_ls, batch_ys = figer.next_context_batch()
            _, print_loss, print_y = sess.run([train_step, loss, y], feed_dict={context: batch_cs, type_emb: batch_ls, y_: batch_ys})

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
                out_file.write('train: epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                                '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')
            # print ('epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
            #                 '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')

            if i%10 == 0:
                print_loss, print_y = sess.run([loss, y], feed_dict={context: test_cs, type_emb: test_ls, y_: test_ys})
                zero_R_count = 0.0
                NN_count = 0.0
                for j in range(0, len(print_y)):
                    if test_ys[j][0] == 0.0:
                        zero_R_count += 1.0
                    if abs(print_y[j] - test_ys[j][0]) < 0.5:
                        NN_count += 1.0

                zero_R_acc = zero_R_count / float(len(print_y))
                NN_acc = NN_count / float(len(print_y))
                with open('loss_record.txt', 'a') as out_file:
                    out_file.write('test: epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                                    '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')
                # print ('test: epoch = ' + str(i) + '\t lost = ' + str(print_loss) + \
                #                 '\t zero_R_acc = ' + str(zero_R_acc) + '\t NN_acc = ' + str(NN_acc) + '\n')

            if i % 100 == 0:
                model_name = 'model/model_' + str(i) + '.ckpt'
                saver.save(sess, model_name)

def w2v_type_restore():
    sess=tf.Session()
    saver = tf.train.import_meta_graph('model/model_2400.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))
    sess.run(tf.global_variables_initializer())




def temp():
    pass

if __name__ == "__main__":
    # temp()
    context_type_CNN()
