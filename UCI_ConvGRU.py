import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

X_train = np.load("./UCI_data/processed/np_train_x.npy")
X_test = np.load("./UCI_data/processed/np_test_x.npy")
y_train = np.load("./UCI_data/processed/np_train_y.npy")
y_test = np.load("./UCI_data/processed/np_test_y.npy")
print('是否保存模型以便在测试时进行调用：1 是 2 否')
op = int(input('请选择：'))

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])

lr = 0.001
training_iters = training_data_count * 300
batch_size = 1000
num_units_gru = 32
n_classes = 6
min_acc = 0

def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def ConvGRU(xs):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
    conv1 = tf.layers.conv2d(xs, 32, [5, 1], [2, 1], 'valid')
    conv2 = tf.layers.conv2d(conv1, 32, [5, 1], [2, 1], 'valid')
    conv3 = tf.layers.conv2d(conv2, 32, [5, 1], [2, 1], 'valid')
    conv4 = tf.layers.conv2d(conv3, 32, [5, 1], [2, 1], 'valid')
    shape = conv4.get_shape().as_list()
    print('gru input shape: {}'.format(shape))
    flat = tf.reshape(conv4, [-1, shape[1], shape[2] * shape[3]])
    rnn_cell_1 = tf.contrib.rnn.GRUCell(num_units_gru)
    rnn_cell_2 = tf.contrib.rnn.GRUCell(num_units_gru)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell_1, rnn_cell_2])
    init_state = rnn_cell.zero_state(_batch_size, dtype=tf.float32)
    outputs, last_states = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        flat,  # input
        initial_state=init_state,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
    )

    output = tf.layers.dense(outputs[:, -1, :], n_classes, activation=tf.nn.softmax)  # output based on the last output step

    return output

xs = tf.placeholder(tf.float32, [None, n_steps, n_input],name='input')
ys = tf.placeholder(tf.float32, [None, n_classes],name='label')
_batch_size = tf.placeholder(dtype=tf.int32, shape=[])

output = ConvGRU(xs)

# loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(output, 1e-10, 1.0)),
                                         reduction_indices=[1]))  # loss

train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
#     labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1), )[1]
argmax_pred = tf.argmax(output, 1, name="output")
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

train_losses = []
with tf.Session() as sess:
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    step = 1
    start_time = time.time()
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = extract_batch_size(y_train, step, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys, _batch_size: batch_size})
        train_losses.append(loss_)
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, _batch_size: test_data_count})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_
                  , '| time: %.4f'% (time.time() - start_time))
        if step % 100 == 0 and op == 1 and accuracy_>min_acc:
            saver.save(sess, "./ConvLSTMmodel/ConvLSTM_model")
            min_acc = accuracy_
            print('ConvLSTM模型保存成功')
        if step % 1000 == 0:
            indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
            plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
            plt.show()
        step += 1
# kenn
# Step: 50 | train loss: 0.5131 | test accuracy: 0.7418 | time: 5.3166
# Step: 100 | train loss: 0.2256 | test accuracy: 0.8327 | time: 9.2845
# Step: 150 | train loss: 0.1312 | test accuracy: 0.8744 | time: 13.0134
# Step: 200 | train loss: 0.1754 | test accuracy: 0.8839 | time: 16.5852
# Step: 250 | train loss: 0.1542 | test accuracy: 0.8829 | time: 20.1220
# Step: 300 | train loss: 0.0689 | test accuracy: 0.8918 | time: 23.8638
# Step: 350 | train loss: 0.0846 | test accuracy: 0.8924 | time: 27.4306
# Step: 400 | train loss: 0.0683 | test accuracy: 0.8850 | time: 31.1775
# Step: 450 | train loss: 0.1366 | test accuracy: 0.8928 | time: 34.8623
# Step: 500 | train loss: 0.1412 | test accuracy: 0.9023 | time: 38.4451
# Step: 550 | train loss: 0.0430 | test accuracy: 0.9080 | time: 42.1749
# Step: 600 | train loss: 0.0583 | test accuracy: 0.9009 | time: 45.7728
# Step: 650 | train loss: 0.0780 | test accuracy: 0.8979 | time: 49.3826
# Step: 700 | train loss: 0.1185 | test accuracy: 0.9019 | time: 53.0054
# Step: 750 | train loss: 0.1088 | test accuracy: 0.9128 | time: 56.6852
# Step: 800 | train loss: 0.1370 | test accuracy: 0.8975 | time: 60.2550
# Step: 850 | train loss: 0.3081 | test accuracy: 0.8972 | time: 63.9578
# Step: 900 | train loss: 0.0626 | test accuracy: 0.9030 | time: 67.6697
# Step: 950 | train loss: 0.1208 | test accuracy: 0.9169 | time: 71.2075
# Step: 1000 | train loss: 0.0961 | test accuracy: 0.9135 | time: 74.9063

# kenn
# Step: 50 | train loss: 0.7372 | test accuracy: 0.6888 | time: 5.1848
# Step: 100 | train loss: 0.3681 | test accuracy: 0.8103 | time: 8.8166
# Step: 150 | train loss: 0.1912 | test accuracy: 0.8873 | time: 12.5245
# Step: 200 | train loss: 0.1535 | test accuracy: 0.8921 | time: 16.2343
# Step: 250 | train loss: 0.1639 | test accuracy: 0.8833 | time: 19.8781
# Step: 300 | train loss: 0.0753 | test accuracy: 0.8931 | time: 23.6020
# Step: 350 | train loss: 0.0905 | test accuracy: 0.8887 | time: 27.2048
# Step: 400 | train loss: 0.0799 | test accuracy: 0.9013 | time: 30.8656
# Step: 450 | train loss: 0.1327 | test accuracy: 0.8968 | time: 34.3934
# Step: 500 | train loss: 0.1498 | test accuracy: 0.8989 | time: 37.9902
# Step: 550 | train loss: 0.0540 | test accuracy: 0.9057 | time: 41.5660
# Step: 600 | train loss: 0.0668 | test accuracy: 0.9063 | time: 45.2018
# Step: 650 | train loss: 0.0720 | test accuracy: 0.9019 | time: 48.7606
# Step: 700 | train loss: 0.1230 | test accuracy: 0.9135 | time: 52.2314
# Step: 750 | train loss: 0.1714 | test accuracy: 0.9013 | time: 55.8022
# Step: 800 | train loss: 0.0769 | test accuracy: 0.8935 | time: 59.2620
# Step: 850 | train loss: 0.0626 | test accuracy: 0.9097 | time: 62.8118
# Step: 900 | train loss: 0.0669 | test accuracy: 0.9145 | time: 66.4946
# Step: 950 | train loss: 0.1115 | test accuracy: 0.9172 | time: 70.1584
# Step: 1000 | train loss: 0.1445 | test accuracy: 0.9179 | time: 73.9683

# Retro
# Step: 50 | train loss: 0.6411 | test accuracy: 0.7631 | time: 43.4007
# Step: 100 | train loss: 0.2669 | test accuracy: 0.8381 | time: 86.4781
# Step: 150 | train loss: 0.1810 | test accuracy: 0.8677 | time: 128.7609
# Step: 200 | train loss: 0.1945 | test accuracy: 0.8806 | time: 170.2442
# Step: 250 | train loss: 0.1566 | test accuracy: 0.8867 | time: 211.5053
# Step: 300 | train loss: 0.0573 | test accuracy: 0.8914 | time: 254.1954
# Step: 350 | train loss: 0.0731 | test accuracy: 0.8938 | time: 296.8175
# Step: 400 | train loss: 0.0704 | test accuracy: 0.9030 | time: 339.5356
# Step: 450 | train loss: 0.1274 | test accuracy: 0.8958 | time: 381.9896
# Step: 500 | train loss: 0.1421 | test accuracy: 0.8938 | time: 424.4165
# Step: 550 | train loss: 0.0343 | test accuracy: 0.8979 | time: 465.4945
# Step: 600 | train loss: 0.0406 | test accuracy: 0.8941 | time: 507.6002
# Step: 650 | train loss: 0.0494 | test accuracy: 0.9013 | time: 549.6919
# Step: 700 | train loss: 0.1097 | test accuracy: 0.9013 | time: 591.0731
# Step: 750 | train loss: 0.1503 | test accuracy: 0.8948 | time: 645.8688
# Step: 800 | train loss: 0.0248 | test accuracy: 0.9016 | time: 687.8604
# Step: 850 | train loss: 0.0111 | test accuracy: 0.9023 | time: 728.9634
# Step: 900 | train loss: 0.0433 | test accuracy: 0.8951 | time: 770.2616
# Step: 950 | train loss: 0.0704 | test accuracy: 0.8979 | time: 811.3076
# Step: 1000 | train loss: 0.0447 | test accuracy: 0.8778 | time: 855.0184
