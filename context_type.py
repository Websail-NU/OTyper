import tensorflow as tf
import numpy as np
import data_utility


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
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        figer = data_utility.figer_data()
        test_cs, test_ls, test_ys = figer.get_context_test_data(25000, 10000)

        saver = tf.train.Saver()
        for i in range(0,100000):
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
    context_type()
