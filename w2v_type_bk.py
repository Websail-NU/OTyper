import tensorflow as tf
import numpy as np
import data_utility

def w2v_type_linear():
    x = tf.placeholder(tf.float32, [None, 600], name = 'x')
    example_w = tf.placeholder(tf.float32, [None, 1], name = 'example_w')
    W = tf.Variable(tf.zeros([600, 1]), name = 'W')
    b = tf.Variable(tf.zeros([1]), name = 'b')
    y_logit = tf.add(tf.matmul(x, W), b, name = 'y_logit')
    y = tf.sigmoid(y_logit, name = 'y')
    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y_')

    # loss = tf.reduce_mean((y - y_) * (y - y_), name = 'loss')
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_logit, example_w))

    train_step = tf.train.GradientDescentOptimizer(0.33).minimize(loss, name = 'train_step')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    figer = data_utility.figer_data()
    test_xs, test_ys, test_ws = figer.get_test_data(0, 1376897)

    saver = tf.train.Saver()
    for i in range(0,10000):
        batch_xs, batch_ys, batch_ws = figer.next_batch()
        _, print_loss, print_y = sess.run([train_step, loss, y], feed_dict={x: batch_xs, y_: batch_ys, example_w: batch_ws})


        zero_R_count = 0.0
        NN_count = 0.0
        TP = 0.0
        FN = 0.0
        FP = 0.0
        TN = 0.0
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

        with open('loss_record.txt', 'a') as out_file:
            out_file.write('train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                    .format(i, print_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))

        print 'train: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
                .format(i, print_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)


        if i%10 == 0:
            print_loss, print_y = sess.run([loss, y], feed_dict={x: test_xs, y_: test_ys, example_w: test_ws})
            zero_R_count = 0.0
            TP = 0.0
            FN = 0.0
            FP = 0.0
            TN = 0.0
            for j in range(0, len(print_y)):
                if test_ys[j][0] == 0.0:
                    zero_R_count += 1.0
                if test_ys[j][0] == 1.0:
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

            with open('loss_record.txt', 'a') as out_file:
                out_file.write('test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}\n' \
                        .format(i, print_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN))
            print 'test: epoch= {}  loss= {}  zero_R_prec= {}  NN_prec= {}  NN_rec= {}  NN_F1= {}  TP= {}  FN= {}  FP= {}  TN= {}' \
                    .format(i, print_loss, zero_R_prec, NN_prec, NN_rec, NN_F1, TP, FN, FP, TN)

        if i % 100 == 0:
            model_name = 'model/model_' + str(i) + '.ckpt'
            saver.save(sess, model_name)

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
    saver = tf.train.import_meta_graph('model/model_2400.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))
    sess.run(tf.global_variables_initializer())




def temp():
    pass

if __name__ == "__main__":
    # temp()
    w2v_type_linear()
