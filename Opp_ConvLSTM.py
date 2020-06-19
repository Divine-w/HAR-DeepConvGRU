import tensorflow as tf
import numpy as np
from sklearn import metrics
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
X_train = np.load("./Opportunity_data/processed/np_train_x.npy")
X_test = np.load("./Opportunity_data/processed/np_test_x.npy")
y_train = np.load("./Opportunity_data/processed/np_train_y.npy")
y_test = np.load("./Opportunity_data/processed/np_test_y.npy")
argmax_y = y_test.argmax(1)
# print('是否保存模型以便在测试时进行调用：1 是 2 否')
# print('是否保存训练损失：1 是 2 否')
# op = int(input('请选择：'))

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])

lr = 0.001
training_iters = training_data_count * 300
# training_iters = 1000 * 2000
batch_size = 1000
num_units_lstm = 128
n_classes = 18
min_acc = 0
BATCH_SIZE = 100


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def ConvLSTM(xs, is_training):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
    # conv1 = tf.layers.conv2d(xs, 16, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2], padding='same')
    # conv2 = tf.layers.conv2d(pool1, 32, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2], padding='same')
    # conv3 = tf.layers.conv2d(pool2, 64, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool3 = tf.layers.max_pooling2d(conv3, [2, 4], [2, 4], padding='same')
    # shape = pool3.get_shape().as_list()
    # print('lstm input shape: {}'.format(shape))
    # flat = tf.reshape(pool3, [-1, shape[1] * shape[2], shape[3]])
    conv1 = tf.layers.conv2d(xs, 16, [1, 8], 1, 'same', activation=tf.nn.relu)
    conv1 = tf.contrib.layers.batch_norm(inputs=conv1,
                                         decay=0.95,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         updates_collections=None)
    pool1 = tf.layers.max_pooling2d(conv1, [2, 1], [2, 1], padding='same')
    conv2 = tf.layers.conv2d(pool1, 32, [1, 8], 1, 'same', activation=tf.nn.relu)
    conv2 = tf.contrib.layers.batch_norm(inputs=conv2,
                                         decay=0.95,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         updates_collections=None)
    pool2 = tf.layers.max_pooling2d(conv2, [2, 1], [2, 1], padding='same')
    conv3 = tf.layers.conv2d(pool2, 64, [1, 8], 1, 'same', activation=tf.nn.relu)
    conv3 = tf.contrib.layers.batch_norm(inputs=conv3,
                                         decay=0.95,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         updates_collections=None)
    # pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2], padding='same')
    shape = conv3.get_shape().as_list()
    print('lstm input shape: {}'.format(shape))
    flat = tf.reshape(conv3, [-1, shape[1], shape[2] * shape[3]])
    rnn_cell_1 = tf.contrib.rnn.LSTMCell(num_units_lstm)
    rnn_cell_2 = tf.contrib.rnn.LSTMCell(num_units_lstm)
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_1, rnn_cell_2])
    outputs, last_states = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        flat,  # input
        initial_state=None,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
    )

    output = tf.layers.dense(outputs[:, -1, :], n_classes,
                             activation=tf.nn.softmax)  # output based on the last output step

    return output


xs = tf.placeholder(tf.float32, [None, n_steps, n_input], name='input')
ys = tf.placeholder(tf.float32, [None, n_classes], name='label')
is_training = tf.placeholder(tf.bool, name='train')

output = ConvLSTM(xs, is_training)

# loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(output, 1e-10, 1.0)),
                                     reduction_indices=[1]))  # loss

train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
#     labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1), )[1]
argmax_pred = tf.argmax(output, 1, name="output")
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

train_losses = []
Time = []
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer())  # the local var is for accuracy_op
    sess.run(init_op)  # initialize var in graph

    step = 1
    start_time = time.time()
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = extract_batch_size(y_train, step, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys, is_training: True})
        train_losses.append(loss_)
        Time.append(time.time() - start_time)
        if step % 50 == 0:
            test_pred = np.empty((0))
            test_true = np.empty((0))
            for batch in iterate_minibatches(X_test, argmax_y, BATCH_SIZE):
                inputs, targets = batch
                y_pred = sess.run(argmax_pred, feed_dict={xs: inputs, is_training: False})
                test_pred = np.append(test_pred, y_pred, axis=0)
                test_true = np.append(test_true, targets, axis=0)
            # pred = sess.run(argmax_pred, feed_dict={xs: X_test})
            # accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_
                  # , '| f1分数: %.4f' % (100 * metrics.f1_score(argmax_y, pred, average='weighted'))
                  , '| f1分数: %.4f' % (100 * metrics.f1_score(test_true, test_pred, average='weighted'))
                  , '| time: %.4f' % (time.time() - start_time))
            print("精度: {:.4f}%".format(100 * metrics.precision_score(test_true, test_pred, average="weighted")))
            print("召回率: {:.4f}%".format(100 * metrics.recall_score(test_true, test_pred, average="weighted")))
        # if step % 2000 == 0 and op == 1:
        #     np.save("./figure/OPP1/lstm_losses_5.npy", train_losses)
        #     np.save("./figure/OPP1/lstm_time_5.npy", Time)
        #     print('损失已保存为numpy文件')
        # if step % 100 == 0 and op == 1 and accuracy_>min_acc:
        #     saver.save(sess, "./OPPmodel1/ConvLSTM_model")
        #     min_acc = accuracy_
        #     print('ConvLSTM模型保存成功')
        # if step % 1000 == 0:
        #     indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
        #     plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        #     plt.show()
        step += 1

# Step: 50 | train loss: 2.9186 | f1分数: 75.6440 | time: 4.4962
# Step: 100 | train loss: 0.8373 | f1分数: 75.6440 | time: 7.6668
# Step: 150 | train loss: 1.0555 | f1分数: 75.6440 | time: 10.8753
# Step: 200 | train loss: 0.8766 | f1分数: 75.6440 | time: 14.1145
# Step: 250 | train loss: 1.2666 | f1分数: 75.6440 | time: 17.3021
# Step: 300 | train loss: 3.1531 | f1分数: 75.6440 | time: 20.5309
# Step: 350 | train loss: 1.1279 | f1分数: 75.6440 | time: 23.7209
# Step: 400 | train loss: 1.1621 | f1分数: 75.6440 | time: 26.8984
# Step: 450 | train loss: 1.0029 | f1分数: 75.6440 | time: 30.1102
# Step: 500 | train loss: 2.2637 | f1分数: 75.6440 | time: 33.3738
# Step: 550 | train loss: 2.3043 | f1分数: 75.6440 | time: 36.5535
# Step: 600 | train loss: 0.8314 | f1分数: 75.6440 | time: 39.8098
# Step: 650 | train loss: 0.9213 | f1分数: 75.6440 | time: 43.0534
# Step: 700 | train loss: 2.6252 | f1分数: 75.6440 | time: 46.2556
# Step: 750 | train loss: 0.7276 | f1分数: 75.6440 | time: 49.4728
# Step: 800 | train loss: 0.8358 | f1分数: 75.6440 | time: 52.6441
# Step: 850 | train loss: 0.9263 | f1分数: 75.6440 | time: 55.8331
# Step: 900 | train loss: 0.9636 | f1分数: 75.6440 | time: 59.0002
# Step: 950 | train loss: 1.2837 | f1分数: 75.6440 | time: 62.2618
# Step: 1000 | train loss: 2.3215 | f1分数: 75.6440 | time: 65.4449
# Step: 1050 | train loss: 0.6966 | f1分数: 75.6440 | time: 68.6438
# Step: 1100 | train loss: 0.8884 | f1分数: 75.8079 | time: 71.9029
# Step: 1150 | train loss: 1.4286 | f1分数: 76.1738 | time: 75.1786
# Step: 1200 | train loss: 1.3428 | f1分数: 76.7771 | time: 78.4614
# Step: 1250 | train loss: 0.8173 | f1分数: 76.9516 | time: 81.6675
# Step: 1300 | train loss: 1.0844 | f1分数: 77.5119 | time: 84.8563
# Step: 1350 | train loss: 1.2239 | f1分数: 78.0152 | time: 88.0870
# Step: 1400 | train loss: 0.3961 | f1分数: 78.5838 | time: 91.2763
# Step: 1450 | train loss: 0.4802 | f1分数: 78.3260 | time: 94.4505
# Step: 1500 | train loss: 0.4600 | f1分数: 78.8427 | time: 97.5584
# Step: 1550 | train loss: 0.4862 | f1分数: 79.1014 | time: 100.7140
# Step: 1600 | train loss: 0.4008 | f1分数: 79.3375 | time: 103.7786
# Step: 1650 | train loss: 1.2140 | f1分数: 79.3864 | time: 106.9596
# Step: 1700 | train loss: 0.5204 | f1分数: 80.0086 | time: 110.1459
# Step: 1750 | train loss: 0.4867 | f1分数: 80.8503 | time: 113.3411
# Step: 1800 | train loss: 0.7430 | f1分数: 81.6535 | time: 116.5609
# Step: 1850 | train loss: 0.8394 | f1分数: 79.5461 | time: 119.7904
# Step: 1900 | train loss: 0.4219 | f1分数: 81.4858 | time: 122.9467
# Step: 1950 | train loss: 0.4051 | f1分数: 81.7657 | time: 126.2047
# Step: 2000 | train loss: 0.7130 | f1分数: 82.1033 | time: 129.3936
# Step: 2050 | train loss: 0.8198 | f1分数: 82.6387 | time: 132.6100
# Step: 2100 | train loss: 0.2773 | f1分数: 81.5941 | time: 135.8632
# Step: 2150 | train loss: 0.4888 | f1分数: 81.0775 | time: 139.0672
# Step: 2200 | train loss: 0.4834 | f1分数: 81.1932 | time: 142.3436
# Step: 2250 | train loss: 0.4310 | f1分数: 81.8533 | time: 145.5781
# Step: 2300 | train loss: 0.8578 | f1分数: 84.0645 | time: 148.8284
# Step: 2350 | train loss: 0.3063 | f1分数: 85.3034 | time: 152.0050
# Step: 2400 | train loss: 0.4705 | f1分数: 85.1571 | time: 155.2124
# Step: 2450 | train loss: 0.2791 | f1分数: 84.9737 | time: 158.4037
# Step: 2500 | train loss: 0.5971 | f1分数: 83.5518 | time: 161.6016
# Step: 2550 | train loss: 0.2670 | f1分数: 81.2940 | time: 164.7974
# Step: 2600 | train loss: 0.3049 | f1分数: 79.0886 | time: 168.0343
# Step: 2650 | train loss: 0.5363 | f1分数: 83.6915 | time: 171.2748
# Step: 2700 | train loss: 0.6306 | f1分数: 84.2991 | time: 174.4952
# Step: 2750 | train loss: 0.2457 | f1分数: 82.4220 | time: 177.7515
# Step: 2800 | train loss: 0.2960 | f1分数: 81.4597 | time: 181.0017
# Step: 2850 | train loss: 0.3941 | f1分数: 81.7923 | time: 184.2172
# Step: 2900 | train loss: 0.3712 | f1分数: 82.6475 | time: 187.4747
# Step: 2950 | train loss: 0.7832 | f1分数: 84.0475 | time: 190.6847
# Step: 3000 | train loss: 0.3381 | f1分数: 85.4439 | time: 193.8925
# Step: 3050 | train loss: 0.4417 | f1分数: 85.0732 | time: 197.0985
# Step: 3100 | train loss: 0.4346 | f1分数: 85.9124 | time: 200.2946
# Step: 3150 | train loss: 0.4284 | f1分数: 85.8645 | time: 203.4476
# Step: 3200 | train loss: 0.4624 | f1分数: 85.5977 | time: 206.6167
# Step: 3250 | train loss: 0.3939 | f1分数: 85.1391 | time: 209.8079
# Step: 3300 | train loss: 0.2985 | f1分数: 83.7165 | time: 213.0133
# Step: 3350 | train loss: 0.5020 | f1分数: 84.2264 | time: 216.2705
# Step: 3400 | train loss: 0.3199 | f1分数: 84.4117 | time: 219.4208
# Step: 3450 | train loss: 0.3941 | f1分数: 83.4263 | time: 222.6419
# Step: 3500 | train loss: 0.2729 | f1分数: 81.1025 | time: 225.8580
# Step: 3550 | train loss: 0.2740 | f1分数: 82.2621 | time: 229.1237
# Step: 3600 | train loss: 0.2796 | f1分数: 83.3543 | time: 232.3573
# Step: 3650 | train loss: 0.4704 | f1分数: 84.8706 | time: 235.5457
# Step: 3700 | train loss: 0.2121 | f1分数: 85.0978 | time: 238.7599
# Step: 3750 | train loss: 0.3600 | f1分数: 85.0568 | time: 241.9462
# Step: 3800 | train loss: 0.6069 | f1分数: 85.6894 | time: 245.1533
# Step: 3850 | train loss: 0.3511 | f1分数: 85.2127 | time: 248.3146
# Step: 3900 | train loss: 0.4465 | f1分数: 86.9837 | time: 251.5163
# Step: 3950 | train loss: 0.4330 | f1分数: 85.1540 | time: 254.6899
# Step: 4000 | train loss: 0.5067 | f1分数: 83.9142 | time: 257.8664
# Step: 4050 | train loss: 0.2036 | f1分数: 84.8683 | time: 261.0464
# Step: 4100 | train loss: 0.1835 | f1分数: 83.2744 | time: 264.2153
# Step: 4150 | train loss: 0.2753 | f1分数: 80.0195 | time: 267.4231
# Step: 4200 | train loss: 0.2958 | f1分数: 81.3529 | time: 270.6193
# Step: 4250 | train loss: 0.1219 | f1分数: 83.1222 | time: 273.8398
# Step: 4300 | train loss: 0.4027 | f1分数: 84.9287 | time: 277.0772
# Step: 4350 | train loss: 0.2135 | f1分数: 86.0687 | time: 280.2545
# Step: 4400 | train loss: 0.2570 | f1分数: 85.5762 | time: 283.3846
# Step: 4450 | train loss: 0.1868 | f1分数: 86.3812 | time: 286.5650
# Step: 4500 | train loss: 0.3051 | f1分数: 86.8234 | time: 289.7557
# Step: 4550 | train loss: 0.1704 | f1分数: 83.1646 | time: 292.9417
# Step: 4600 | train loss: 0.2012 | f1分数: 86.3617 | time: 296.1415
# Step: 4650 | train loss: 0.4204 | f1分数: 86.7339 | time: 299.3439
# Step: 4700 | train loss: 0.3366 | f1分数: 86.6063 | time: 302.5887
# Step: 4750 | train loss: 0.2203 | f1分数: 85.3647 | time: 305.7693
# Step: 4800 | train loss: 0.2157 | f1分数: 84.6089 | time: 308.9887
# Step: 4850 | train loss: 0.1778 | f1分数: 83.4629 | time: 312.1764
# Step: 4900 | train loss: 0.1947 | f1分数: 82.3447 | time: 315.3451
# Step: 4950 | train loss: 0.3467 | f1分数: 85.6475 | time: 318.5383
# Step: 5000 | train loss: 0.1960 | f1分数: 85.0793 | time: 321.7023
# Step: 5050 | train loss: 0.1779 | f1分数: 85.6573 | time: 324.8804
# Step: 5100 | train loss: 0.2469 | f1分数: 86.7607 | time: 328.0984
# Step: 5150 | train loss: 0.2695 | f1分数: 86.8400 | time: 331.2558
# Step: 5200 | train loss: 0.2118 | f1分数: 86.0818 | time: 334.4401
# Step: 5250 | train loss: 0.2083 | f1分数: 86.2338 | time: 337.6380
# Step: 5300 | train loss: 0.2465 | f1分数: 87.1933 | time: 340.7953
# Step: 5350 | train loss: 0.2948 | f1分数: 87.4178 | time: 343.9957
# Step: 5400 | train loss: 0.2331 | f1分数: 87.8495 | time: 347.2245
# Step: 5450 | train loss: 0.3233 | f1分数: 87.1823 | time: 350.4426
# Step: 5500 | train loss: 0.1872 | f1分数: 86.5614 | time: 353.6584
# Step: 5550 | train loss: 0.2468 | f1分数: 85.5875 | time: 356.8089
# Step: 5600 | train loss: 0.3139 | f1分数: 86.8341 | time: 360.0113
# Step: 5650 | train loss: 0.1977 | f1分数: 85.6145 | time: 363.2055
# Step: 5700 | train loss: 0.2803 | f1分数: 85.9112 | time: 366.4094
# Step: 5750 | train loss: 0.1809 | f1分数: 87.5793 | time: 369.6119
# Step: 5800 | train loss: 0.2089 | f1分数: 87.8370 | time: 372.8118
# Step: 5850 | train loss: 0.1859 | f1分数: 87.6363 | time: 376.0126
# Step: 5900 | train loss: 0.2586 | f1分数: 87.4735 | time: 379.1974
# Step: 5950 | train loss: 0.1239 | f1分数: 86.4398 | time: 382.3882
# Step: 6000 | train loss: 0.2981 | f1分数: 87.4510 | time: 385.5645
# Step: 6050 | train loss: 0.1394 | f1分数: 86.6220 | time: 388.7563
# Step: 6100 | train loss: 0.2759 | f1分数: 86.5302 | time: 392.0063
# Step: 6150 | train loss: 0.1519 | f1分数: 86.5834 | time: 395.1871
# Step: 6200 | train loss: 0.1149 | f1分数: 86.6352 | time: 398.3837
# Step: 6250 | train loss: 0.1132 | f1分数: 86.9414 | time: 401.6696
# Step: 6300 | train loss: 0.2212 | f1分数: 87.8998 | time: 404.9448
# Step: 6350 | train loss: 0.1994 | f1分数: 86.7943 | time: 408.1830
# Step: 6400 | train loss: 0.1258 | f1分数: 87.7167 | time: 411.4335
# Step: 6450 | train loss: 0.2497 | f1分数: 87.4551 | time: 414.6260
# Step: 6500 | train loss: 0.1858 | f1分数: 87.0111 | time: 417.8265
# Step: 6550 | train loss: 0.2386 | f1分数: 86.3305 | time: 421.0242
# Step: 6600 | train loss: 0.2375 | f1分数: 87.9979 | time: 424.2420
# Step: 6650 | train loss: 0.2131 | f1分数: 86.8233 | time: 427.4700
# Step: 6700 | train loss: 0.1146 | f1分数: 84.9784 | time: 430.7223
# Step: 6750 | train loss: 0.1260 | f1分数: 85.0337 | time: 433.9143
# Step: 6800 | train loss: 0.1541 | f1分数: 85.8104 | time: 437.0998
# Step: 6850 | train loss: 0.1319 | f1分数: 86.6047 | time: 440.2252
# Step: 6900 | train loss: 0.1059 | f1分数: 86.4175 | time: 443.4855
# Step: 6950 | train loss: 0.1914 | f1分数: 87.2053 | time: 446.7366
# Step: 7000 | train loss: 0.1405 | f1分数: 85.0497 | time: 449.8974
# Step: 7050 | train loss: 0.1804 | f1分数: 86.9417 | time: 453.1263
# Step: 7100 | train loss: 0.0924 | f1分数: 87.6196 | time: 456.3609
# Step: 7150 | train loss: 0.1692 | f1分数: 87.8329 | time: 459.4808
# Step: 7200 | train loss: 0.0820 | f1分数: 87.5504 | time: 462.6051
# Step: 7250 | train loss: 0.0772 | f1分数: 86.9564 | time: 465.6993
# Step: 7300 | train loss: 0.1761 | f1分数: 87.5068 | time: 468.8480
# Step: 7350 | train loss: 0.1601 | f1分数: 84.1105 | time: 472.0739
# Step: 7400 | train loss: 0.1117 | f1分数: 86.4882 | time: 475.2928
# Step: 7450 | train loss: 0.1133 | f1分数: 83.3243 | time: 478.4801
# Step: 7500 | train loss: 0.1043 | f1分数: 86.8255 | time: 481.6519
# Step: 7550 | train loss: 0.0886 | f1分数: 87.0837 | time: 484.7977
# Step: 7600 | train loss: 0.1984 | f1分数: 87.3622 | time: 487.9646
# Step: 7650 | train loss: 0.0939 | f1分数: 88.2871 | time: 491.1874
# Step: 7700 | train loss: 0.1010 | f1分数: 88.4141 | time: 494.4174
# Step: 7750 | train loss: 0.1288 | f1分数: 88.0815 | time: 497.5969
# Step: 7800 | train loss: 0.1617 | f1分数: 84.7270 | time: 500.8217
# Step: 7850 | train loss: 0.1259 | f1分数: 88.1477 | time: 504.0124
# Step: 7900 | train loss: 0.1287 | f1分数: 87.9146 | time: 507.2228
# Step: 7950 | train loss: 0.0983 | f1分数: 87.8196 | time: 510.4333
# Step: 8000 | train loss: 0.1584 | f1分数: 87.4170 | time: 513.6358
# Step: 8050 | train loss: 0.1920 | f1分数: 86.4614 | time: 516.8582
# Step: 8100 | train loss: 0.1938 | f1分数: 83.5663 | time: 520.0267
# Step: 8150 | train loss: 0.1256 | f1分数: 82.2870 | time: 523.1769
# Step: 8200 | train loss: 0.1656 | f1分数: 84.5077 | time: 526.3207
# Step: 8250 | train loss: 0.2111 | f1分数: 85.4689 | time: 529.5110
# Step: 8300 | train loss: 0.1385 | f1分数: 87.1552 | time: 532.7040
# Step: 8350 | train loss: 0.1701 | f1分数: 87.8895 | time: 535.8740
# Step: 8400 | train loss: 0.0646 | f1分数: 88.0705 | time: 539.0696
# Step: 8450 | train loss: 0.0840 | f1分数: 88.3422 | time: 542.2681
# Step: 8500 | train loss: 0.1310 | f1分数: 87.6513 | time: 545.4374
# Step: 8550 | train loss: 0.1555 | f1分数: 87.6731 | time: 548.6570
# Step: 8600 | train loss: 0.0764 | f1分数: 87.7362 | time: 551.8926
# Step: 8650 | train loss: 0.2866 | f1分数: 87.0083 | time: 555.1165
# Step: 8700 | train loss: 0.0288 | f1分数: 85.6764 | time: 558.2980
# Step: 8750 | train loss: 0.1163 | f1分数: 86.3969 | time: 561.5093
# Step: 8800 | train loss: 0.1033 | f1分数: 85.5026 | time: 564.6644
# Step: 8850 | train loss: 0.0985 | f1分数: 86.4949 | time: 567.8542
# Step: 8900 | train loss: 0.0489 | f1分数: 86.1453 | time: 571.0286
# Step: 8950 | train loss: 0.1471 | f1分数: 86.7455 | time: 574.1732
# Step: 9000 | train loss: 0.0945 | f1分数: 87.3615 | time: 577.3528
# Step: 9050 | train loss: 0.0515 | f1分数: 87.6788 | time: 580.4831
# Step: 9100 | train loss: 0.1309 | f1分数: 88.3878 | time: 583.6822
# Step: 9150 | train loss: 0.1081 | f1分数: 87.7634 | time: 586.8088
# Step: 9200 | train loss: 0.0539 | f1分数: 87.4091 | time: 589.9821
# Step: 9250 | train loss: 0.1364 | f1分数: 87.8824 | time: 593.1658
# Step: 9300 | train loss: 0.1697 | f1分数: 87.4977 | time: 596.3920
# Step: 9350 | train loss: 0.0909 | f1分数: 87.1680 | time: 599.5759
# Step: 9400 | train loss: 0.0567 | f1分数: 85.9912 | time: 602.7891
# Step: 9450 | train loss: 0.0783 | f1分数: 86.6485 | time: 605.9970
# Step: 9500 | train loss: 0.0592 | f1分数: 86.5777 | time: 609.1815
# Step: 9550 | train loss: 0.0639 | f1分数: 85.3600 | time: 612.3839
# Step: 9600 | train loss: 0.1027 | f1分数: 86.5765 | time: 615.5763
# Step: 9650 | train loss: 0.1035 | f1分数: 86.3228 | time: 618.7817
# Step: 9700 | train loss: 0.0780 | f1分数: 86.8543 | time: 621.9647
# Step: 9750 | train loss: 0.0153 | f1分数: 88.0825 | time: 625.1563
# Step: 9800 | train loss: 0.1104 | f1分数: 88.5396 | time: 628.3608
# Step: 9850 | train loss: 0.0432 | f1分数: 88.2555 | time: 631.5479
# Step: 9900 | train loss: 0.0383 | f1分数: 88.4872 | time: 634.7351
# Step: 9950 | train loss: 0.0701 | f1分数: 88.2389 | time: 637.9668
# Step: 10000 | train loss: 0.0826 | f1分数: 88.2371 | time: 641.1918
# Step: 10050 | train loss: 0.0387 | f1分数: 88.1753 | time: 644.3920
# Step: 10100 | train loss: 0.0665 | f1分数: 88.4204 | time: 647.6340
# Step: 10150 | train loss: 0.0430 | f1分数: 87.9727 | time: 650.9014
# Step: 10200 | train loss: 0.0452 | f1分数: 86.9010 | time: 654.1145
# Step: 10250 | train loss: 0.0979 | f1分数: 88.6279 | time: 657.2864
# Step: 10300 | train loss: 0.0416 | f1分数: 88.7371 | time: 660.5052
# Step: 10350 | train loss: 0.1382 | f1分数: 88.6775 | time: 663.6918
# Step: 10400 | train loss: 0.0789 | f1分数: 88.1474 | time: 666.9806
# Step: 10450 | train loss: 0.1063 | f1分数: 88.5685 | time: 670.2266
# Step: 10500 | train loss: 0.0597 | f1分数: 88.9146 | time: 673.4732
# Step: 10550 | train loss: 0.0361 | f1分数: 88.9088 | time: 676.7166
# Step: 10600 | train loss: 0.0204 | f1分数: 87.8494 | time: 679.9556
# Step: 10650 | train loss: 0.0739 | f1分数: 88.6169 | time: 683.1918
# Step: 10700 | train loss: 0.0443 | f1分数: 88.7883 | time: 686.4318
# Step: 10750 | train loss: 0.0682 | f1分数: 82.9415 | time: 689.6232
# Step: 10800 | train loss: 0.0941 | f1分数: 87.9019 | time: 692.7878
# Step: 10850 | train loss: 0.0631 | f1分数: 88.5851 | time: 695.9643
# Step: 10900 | train loss: 0.0707 | f1分数: 89.1444 | time: 699.1432
# Step: 10950 | train loss: 0.0537 | f1分数: 89.0904 | time: 702.3687
# Step: 11000 | train loss: 0.0407 | f1分数: 88.4418 | time: 705.5554
# Step: 11050 | train loss: 0.0288 | f1分数: 87.8238 | time: 708.7507
# Step: 11100 | train loss: 0.0617 | f1分数: 88.1873 | time: 711.9788
# Step: 11150 | train loss: 0.0460 | f1分数: 88.0535 | time: 715.1606
# Step: 11200 | train loss: 0.0553 | f1分数: 88.2666 | time: 718.4549
# Step: 11250 | train loss: 0.0738 | f1分数: 88.3336 | time: 721.6778
# Step: 11300 | train loss: 0.0931 | f1分数: 88.5123 | time: 724.9622
# Step: 11350 | train loss: 0.0088 | f1分数: 88.1156 | time: 728.2343
# Step: 11400 | train loss: 0.0307 | f1分数: 87.8558 | time: 731.4407
# Step: 11450 | train loss: 0.0276 | f1分数: 86.8765 | time: 734.7034
# Step: 11500 | train loss: 0.0292 | f1分数: 86.8430 | time: 737.9176
# Step: 11550 | train loss: 0.0337 | f1分数: 87.4333 | time: 741.1458
# Step: 11600 | train loss: 0.0769 | f1分数: 87.7762 | time: 744.3532
# Step: 11650 | train loss: 0.0255 | f1分数: 87.5256 | time: 747.5332
# Step: 11700 | train loss: 0.0270 | f1分数: 88.0593 | time: 750.7369
# Step: 11750 | train loss: 0.0423 | f1分数: 88.3263 | time: 753.9050
# Step: 11800 | train loss: 0.0370 | f1分数: 88.2407 | time: 757.1217
# Step: 11850 | train loss: 0.0201 | f1分数: 85.5148 | time: 760.3122
# Step: 11900 | train loss: 0.0743 | f1分数: 87.7661 | time: 763.5874
# Step: 11950 | train loss: 0.0678 | f1分数: 88.1976 | time: 766.7770
# Step: 12000 | train loss: 0.0318 | f1分数: 88.1883 | time: 769.9632
# Step: 12050 | train loss: 0.0398 | f1分数: 86.8598 | time: 773.2097
# Step: 12100 | train loss: 0.0691 | f1分数: 86.7631 | time: 776.4494
# Step: 12150 | train loss: 0.0355 | f1分数: 86.7772 | time: 779.6520
# Step: 12200 | train loss: 0.0222 | f1分数: 87.0566 | time: 782.9279
# Step: 12250 | train loss: 0.0490 | f1分数: 87.7891 | time: 786.1310
# Step: 12300 | train loss: 0.0267 | f1分数: 87.9596 | time: 789.3690
# Step: 12350 | train loss: 0.0267 | f1分数: 88.0635 | time: 792.5374
# Step: 12400 | train loss: 0.0116 | f1分数: 88.9484 | time: 795.7764
# Step: 12450 | train loss: 0.0636 | f1分数: 89.0028 | time: 799.0117
# Step: 12500 | train loss: 0.0372 | f1分数: 82.6176 | time: 802.2296
# Step: 12550 | train loss: 0.0191 | f1分数: 89.0205 | time: 805.4185
# Step: 12600 | train loss: 0.0275 | f1分数: 89.0004 | time: 808.6187
# Step: 12650 | train loss: 0.0290 | f1分数: 88.6137 | time: 811.8772
# Step: 12700 | train loss: 0.0136 | f1分数: 87.8899 | time: 815.0678
# Step: 12750 | train loss: 0.0209 | f1分数: 88.1461 | time: 818.3480
# Step: 12800 | train loss: 0.0166 | f1分数: 87.7250 | time: 821.5800
# Step: 12850 | train loss: 0.0313 | f1分数: 88.2130 | time: 824.7584
# Step: 12900 | train loss: 0.0662 | f1分数: 87.8646 | time: 827.9679
# Step: 12950 | train loss: 0.0169 | f1分数: 88.7421 | time: 831.2896
# Step: 13000 | train loss: 0.0852 | f1分数: 87.8816 | time: 834.5756
# Step: 13050 | train loss: 0.0182 | f1分数: 88.6285 | time: 837.7916
# Step: 13100 | train loss: 0.0372 | f1分数: 88.3292 | time: 840.9999
# Step: 13150 | train loss: 0.0240 | f1分数: 88.5423 | time: 844.2189
# Step: 13200 | train loss: 0.0155 | f1分数: 88.5151 | time: 847.3987
# Step: 13250 | train loss: 0.0179 | f1分数: 88.6920 | time: 850.5855
# Step: 13300 | train loss: 0.0356 | f1分数: 88.1440 | time: 853.8423
# Step: 13350 | train loss: 0.0278 | f1分数: 87.9784 | time: 857.0805
# Step: 13400 | train loss: 0.0237 | f1分数: 88.7334 | time: 860.2736
# Step: 13450 | train loss: 0.0287 | f1分数: 88.5419 | time: 863.5087
# Step: 13500 | train loss: 0.0326 | f1分数: 87.5590 | time: 866.7756
# Step: 13550 | train loss: 0.0410 | f1分数: 88.4672 | time: 869.9642
# Step: 13600 | train loss: 0.0497 | f1分数: 87.4373 | time: 873.1385
# Step: 13650 | train loss: 0.0072 | f1分数: 88.6624 | time: 876.3444
# Step: 13700 | train loss: 0.0250 | f1分数: 88.4935 | time: 879.5487
# Step: 13750 | train loss: 0.0757 | f1分数: 88.5813 | time: 882.7337
# Step: 13800 | train loss: 0.0176 | f1分数: 88.9061 | time: 885.9346
# Step: 13850 | train loss: 0.0105 | f1分数: 88.2129 | time: 889.1381
# Step: 13900 | train loss: 0.0090 | f1分数: 89.2008 | time: 892.3212

# padding = same
# Step: 50 | train loss: 2.8608 | f1分数: 75.6440 | time: 3.9145
# Step: 100 | train loss: 0.8371 | f1分数: 75.6440 | time: 6.6406
# Step: 150 | train loss: 1.0572 | f1分数: 75.6440 | time: 9.4027
# Step: 200 | train loss: 0.8746 | f1分数: 75.6440 | time: 12.1540
# Step: 250 | train loss: 1.2318 | f1分数: 75.6440 | time: 14.9155
# Step: 300 | train loss: 2.8895 | f1分数: 75.6440 | time: 17.6412
# Step: 350 | train loss: 0.9603 | f1分数: 75.6440 | time: 20.3917
# Step: 400 | train loss: 1.3149 | f1分数: 75.6440 | time: 23.1624
# Step: 450 | train loss: 0.7428 | f1分数: 75.6440 | time: 25.9061
# Step: 500 | train loss: 1.8265 | f1分数: 76.2572 | time: 28.6849
# Step: 550 | train loss: 1.4922 | f1分数: 77.1890 | time: 31.4521
# Step: 600 | train loss: 0.5630 | f1分数: 77.8253 | time: 34.2141
# Step: 650 | train loss: 0.6065 | f1分数: 78.8016 | time: 36.9669
# Step: 700 | train loss: 1.0850 | f1分数: 79.4570 | time: 39.7005
# Step: 750 | train loss: 0.3739 | f1分数: 78.6140 | time: 42.5111
# Step: 800 | train loss: 0.6300 | f1分数: 78.9738 | time: 45.2766
# Step: 850 | train loss: 0.5863 | f1分数: 79.7780 | time: 47.9850
# Step: 900 | train loss: 0.7107 | f1分数: 79.8878 | time: 50.7516
# Step: 950 | train loss: 0.6243 | f1分数: 81.0065 | time: 53.4939
# Step: 1000 | train loss: 0.7161 | f1分数: 83.5495 | time: 56.2325
# Step: 1050 | train loss: 0.4524 | f1分数: 83.3469 | time: 59.0159
# Step: 1100 | train loss: 0.5290 | f1分数: 83.8293 | time: 61.7515
# Step: 1150 | train loss: 0.5330 | f1分数: 84.7056 | time: 64.5037
# Step: 1200 | train loss: 0.5194 | f1分数: 83.0799 | time: 67.2553
# Step: 1250 | train loss: 0.6188 | f1分数: 84.3117 | time: 70.0677
# Step: 1300 | train loss: 0.6572 | f1分数: 84.3467 | time: 72.8449
# Step: 1350 | train loss: 0.5484 | f1分数: 83.5905 | time: 75.5888
# Step: 1400 | train loss: 0.1145 | f1分数: 83.2386 | time: 78.3810
# Step: 1450 | train loss: 0.2516 | f1分数: 81.6228 | time: 81.1458
# Step: 1500 | train loss: 0.4164 | f1分数: 81.5572 | time: 83.9201
# Step: 1550 | train loss: 0.3534 | f1分数: 81.7174 | time: 86.7003
# Step: 1600 | train loss: 0.2691 | f1分数: 83.2071 | time: 89.4216
# Step: 1650 | train loss: 0.4541 | f1分数: 84.6274 | time: 92.2059
# Step: 1700 | train loss: 0.2896 | f1分数: 85.7534 | time: 94.9742
# Step: 1750 | train loss: 0.3140 | f1分数: 86.0644 | time: 97.7233
# Step: 1800 | train loss: 0.3619 | f1分数: 85.6938 | time: 100.4633
# Step: 1850 | train loss: 0.3003 | f1分数: 86.9069 | time: 103.2320
# Step: 1900 | train loss: 0.2388 | f1分数: 85.8404 | time: 105.9795
# Step: 1950 | train loss: 0.2864 | f1分数: 85.6567 | time: 108.7459
# Step: 2000 | train loss: 0.3575 | f1分数: 86.5257 | time: 111.5060
# Step: 2050 | train loss: 0.3575 | f1分数: 85.4222 | time: 114.2842
# Step: 2100 | train loss: 0.2606 | f1分数: 85.6659 | time: 117.0541
# Step: 2150 | train loss: 0.2864 | f1分数: 85.6455 | time: 119.7974
# Step: 2200 | train loss: 0.2485 | f1分数: 85.0398 | time: 122.5442
# Step: 2250 | train loss: 0.2413 | f1分数: 85.3599 | time: 125.3202
# Step: 2300 | train loss: 0.2953 | f1分数: 86.1309 | time: 128.0562
# Step: 2350 | train loss: 0.1846 | f1分数: 86.8623 | time: 130.8396
# Step: 2400 | train loss: 0.2620 | f1分数: 86.9911 | time: 133.5958
# Step: 2450 | train loss: 0.1508 | f1分数: 87.3768 | time: 136.3543
# Step: 2500 | train loss: 0.2012 | f1分数: 87.3818 | time: 139.1010
# Step: 2550 | train loss: 0.0847 | f1分数: 86.4325 | time: 141.8849
# Step: 2600 | train loss: 0.0953 | f1分数: 83.7947 | time: 144.6297
# Step: 2650 | train loss: 0.2464 | f1分数: 86.0836 | time: 147.3550
# Step: 2700 | train loss: 0.2323 | f1分数: 85.6627 | time: 150.0988
# Step: 2750 | train loss: 0.0730 | f1分数: 86.1653 | time: 152.8474
# Step: 2800 | train loss: 0.2241 | f1分数: 85.5666 | time: 155.6108
# Step: 2850 | train loss: 0.1777 | f1分数: 83.2800 | time: 158.3781
# Step: 2900 | train loss: 0.1854 | f1分数: 85.9739 | time: 161.1341
# Step: 2950 | train loss: 0.2941 | f1分数: 81.5385 | time: 163.8972
# Step: 3000 | train loss: 0.0791 | f1分数: 86.4609 | time: 166.6809
# Step: 3050 | train loss: 0.1959 | f1分数: 86.8403 | time: 169.4483
# Step: 3100 | train loss: 0.1504 | f1分数: 87.9906 | time: 172.2336
# Step: 3150 | train loss: 0.1410 | f1分数: 87.8214 | time: 175.0333
# Step: 3200 | train loss: 0.1320 | f1分数: 87.6229 | time: 177.7934
# Step: 3250 | train loss: 0.1335 | f1分数: 86.8440 | time: 180.5389
# Step: 3300 | train loss: 0.1274 | f1分数: 87.6020 | time: 183.2919
# Step: 3350 | train loss: 0.1907 | f1分数: 88.5329 | time: 186.0616
# Step: 3400 | train loss: 0.0950 | f1分数: 88.4380 | time: 188.8200
# Step: 3450 | train loss: 0.1376 | f1分数: 86.0890 | time: 191.5928
# Step: 3500 | train loss: 0.2687 | f1分数: 87.1459 | time: 194.3585
# Step: 3550 | train loss: 0.2029 | f1分数: 87.2565 | time: 197.1126
# Step: 3600 | train loss: 0.0679 | f1分数: 88.9952 | time: 199.8606
# Step: 3650 | train loss: 0.1129 | f1分数: 88.4357 | time: 202.6315
# Step: 3700 | train loss: 0.0707 | f1分数: 88.4316 | time: 205.3899
# Step: 3750 | train loss: 0.0540 | f1分数: 88.4648 | time: 208.1601
# Step: 3800 | train loss: 0.1412 | f1分数: 88.0860 | time: 210.9299
# Step: 3850 | train loss: 0.0683 | f1分数: 87.9993 | time: 213.6677
# Step: 3900 | train loss: 0.0891 | f1分数: 88.4182 | time: 216.4274
# Step: 3950 | train loss: 0.0872 | f1分数: 88.1193 | time: 219.1856
# Step: 4000 | train loss: 0.1323 | f1分数: 84.4532 | time: 221.9640
# Step: 4050 | train loss: 0.0305 | f1分数: 88.3970 | time: 224.7086
# Step: 4100 | train loss: 0.0708 | f1分数: 88.6736 | time: 227.4909
# Step: 4150 | train loss: 0.0746 | f1分数: 86.1185 | time: 230.2630
# Step: 4200 | train loss: 0.0402 | f1分数: 88.4720 | time: 232.9981
# Step: 4250 | train loss: 0.0470 | f1分数: 89.2133 | time: 235.7527
# Step: 4300 | train loss: 0.0806 | f1分数: 89.0191 | time: 238.5487
# Step: 4350 | train loss: 0.0293 | f1分数: 88.3634 | time: 241.2720
# Step: 4400 | train loss: 0.0521 | f1分数: 88.4467 | time: 244.0313
# Step: 4450 | train loss: 0.0303 | f1分数: 88.5925 | time: 246.8172
# Step: 4500 | train loss: 0.0749 | f1分数: 88.0401 | time: 249.5883
# Step: 4550 | train loss: 0.0243 | f1分数: 88.6076 | time: 252.3757
# Step: 4600 | train loss: 0.0353 | f1分数: 87.4019 | time: 255.1231
# Step: 4650 | train loss: 0.0512 | f1分数: 88.1413 | time: 257.8784
# Step: 4700 | train loss: 0.0433 | f1分数: 88.2409 | time: 260.6526
# Step: 4750 | train loss: 0.0499 | f1分数: 88.0825 | time: 263.3953
# Step: 4800 | train loss: 0.0937 | f1分数: 87.7956 | time: 266.1463
# Step: 4850 | train loss: 0.0114 | f1分数: 88.1583 | time: 268.8997
# Step: 4900 | train loss: 0.0244 | f1分数: 86.7155 | time: 271.6820
# Step: 4950 | train loss: 0.0699 | f1分数: 87.0321 | time: 274.4304
# Step: 5000 | train loss: 0.0272 | f1分数: 87.8532 | time: 277.1981
# Step: 5050 | train loss: 0.0117 | f1分数: 87.6765 | time: 279.9518
# Step: 5100 | train loss: 0.0478 | f1分数: 88.3741 | time: 282.6926
# Step: 5150 | train loss: 0.0674 | f1分数: 88.3344 | time: 285.4786
# Step: 5200 | train loss: 0.0247 | f1分数: 88.3737 | time: 288.2706
# Step: 5250 | train loss: 0.0071 | f1分数: 88.6746 | time: 291.0549
# Step: 5300 | train loss: 0.0411 | f1分数: 88.2826 | time: 293.8076
# Step: 5350 | train loss: 0.0420 | f1分数: 89.0039 | time: 296.5785
# Step: 5400 | train loss: 0.0312 | f1分数: 88.5243 | time: 299.3576
# Step: 5450 | train loss: 0.0674 | f1分数: 87.3338 | time: 302.1332
# Step: 5500 | train loss: 0.0540 | f1分数: 87.4707 | time: 304.8915
# Step: 5550 | train loss: 0.0691 | f1分数: 87.3926 | time: 307.6383
# Step: 5600 | train loss: 0.0454 | f1分数: 87.4880 | time: 310.3950
# Step: 5650 | train loss: 0.0241 | f1分数: 87.2897 | time: 313.1515
# Step: 5700 | train loss: 0.1917 | f1分数: 88.0938 | time: 315.9213
# Step: 5750 | train loss: 0.0180 | f1分数: 88.7458 | time: 318.6967
# Step: 5800 | train loss: 0.0526 | f1分数: 88.7397 | time: 321.4492
# Step: 5850 | train loss: 0.0236 | f1分数: 88.9028 | time: 324.2200
# Step: 5900 | train loss: 0.0250 | f1分数: 88.8772 | time: 327.0699
# Step: 5950 | train loss: 0.0067 | f1分数: 88.4357 | time: 329.8206
# Step: 6000 | train loss: 0.0402 | f1分数: 88.8927 | time: 332.6581
# Step: 6050 | train loss: 0.0150 | f1分数: 89.1110 | time: 335.4552
# Step: 6100 | train loss: 0.0075 | f1分数: 88.8670 | time: 338.2582
# Step: 6150 | train loss: 0.0251 | f1分数: 88.2400 | time: 341.0003
# Step: 6200 | train loss: 0.0440 | f1分数: 88.1801 | time: 343.7908
# Step: 6250 | train loss: 0.0059 | f1分数: 83.4655 | time: 346.5114
# Step: 6300 | train loss: 0.0381 | f1分数: 88.0469 | time: 349.3455
# Step: 6350 | train loss: 0.0125 | f1分数: 88.5007 | time: 352.1057
# Step: 6400 | train loss: 0.0925 | f1分数: 89.0744 | time: 354.9109
# Step: 6450 | train loss: 0.0801 | f1分数: 88.5314 | time: 357.7351
# Step: 6500 | train loss: 0.0403 | f1分数: 88.5472 | time: 360.5971
# Step: 6550 | train loss: 0.0214 | f1分数: 89.0246 | time: 363.4107
# Step: 6600 | train loss: 0.0212 | f1分数: 89.0630 | time: 366.2242
# Step: 6650 | train loss: 0.0774 | f1分数: 88.8693 | time: 369.0874
# Step: 6700 | train loss: 0.0088 | f1分数: 89.1409 | time: 371.9967
# Step: 6750 | train loss: 0.0047 | f1分数: 88.7661 | time: 374.8772
# Step: 6800 | train loss: 0.0352 | f1分数: 88.8523 | time: 377.7348
# Step: 6850 | train loss: 0.0038 | f1分数: 88.8788 | time: 380.5981
# Step: 6900 | train loss: 0.0139 | f1分数: 88.8239 | time: 383.4340
# Step: 6950 | train loss: 0.0396 | f1分数: 88.7393 | time: 386.3208
# Step: 7000 | train loss: 0.0022 | f1分数: 88.5692 | time: 389.2132
# Step: 7050 | train loss: 0.0025 | f1分数: 88.9412 | time: 392.0445
# Step: 7100 | train loss: 0.0017 | f1分数: 88.0624 | time: 394.8447
# Step: 7150 | train loss: 0.0475 | f1分数: 89.0785 | time: 397.6730
# Step: 7200 | train loss: 0.0137 | f1分数: 88.9773 | time: 400.5058
# Step: 7250 | train loss: 0.0039 | f1分数: 88.8442 | time: 403.3506
# Step: 7300 | train loss: 0.0144 | f1分数: 88.6396 | time: 406.1992
# Step: 7350 | train loss: 0.0134 | f1分数: 88.4060 | time: 409.0611
# Step: 7400 | train loss: 0.0132 | f1分数: 89.2476 | time: 411.8789
# Step: 7450 | train loss: 0.0246 | f1分数: 88.7615 | time: 414.7088
# Step: 7500 | train loss: 0.0686 | f1分数: 89.2916 | time: 417.5791
# Step: 7550 | train loss: 0.0069 | f1分数: 87.9420 | time: 420.4165
# Step: 7600 | train loss: 0.0431 | f1分数: 87.5848 | time: 423.2386
# Step: 7650 | train loss: 0.0170 | f1分数: 87.9299 | time: 426.0509
# Step: 7700 | train loss: 0.0033 | f1分数: 88.5514 | time: 428.9156
# Step: 7750 | train loss: 0.0149 | f1分数: 88.6133 | time: 431.7352
# Step: 7800 | train loss: 0.0416 | f1分数: 89.0219 | time: 434.5704
# Step: 7850 | train loss: 0.0271 | f1分数: 88.2541 | time: 437.3825
# Step: 7900 | train loss: 0.0054 | f1分数: 87.4169 | time: 440.1794
# Step: 7950 | train loss: 0.0214 | f1分数: 88.2742 | time: 442.9951
# Step: 8000 | train loss: 0.0310 | f1分数: 88.5346 | time: 445.8200
# Step: 8050 | train loss: 0.0059 | f1分数: 88.1667 | time: 448.6477
# Step: 8100 | train loss: 0.0112 | f1分数: 88.5731 | time: 451.4992
# Step: 8150 | train loss: 0.0251 | f1分数: 88.2539 | time: 454.3402
# Step: 8200 | train loss: 0.0545 | f1分数: 89.1324 | time: 457.2065
# Step: 8250 | train loss: 0.0083 | f1分数: 88.6733 | time: 460.0950
# Step: 8300 | train loss: 0.0094 | f1分数: 88.3964 | time: 462.9456
# Step: 8350 | train loss: 0.0155 | f1分数: 88.8218 | time: 465.7818
# Step: 8400 | train loss: 0.0017 | f1分数: 88.9039 | time: 468.5958
# Step: 8450 | train loss: 0.0176 | f1分数: 88.6650 | time: 471.4803
# Step: 8500 | train loss: 0.0062 | f1分数: 88.6768 | time: 474.3355
# Step: 8550 | train loss: 0.0334 | f1分数: 88.8592 | time: 477.1917
# Step: 8600 | train loss: 0.0014 | f1分数: 89.1706 | time: 480.0626
# Step: 8650 | train loss: 0.0231 | f1分数: 88.8620 | time: 482.9358
# Step: 8700 | train loss: 0.0001 | f1分数: 88.1509 | time: 485.7960
# Step: 8750 | train loss: 0.0129 | f1分数: 88.3700 | time: 488.6805
# Step: 8800 | train loss: 0.0069 | f1分数: 89.1738 | time: 491.5456
# Step: 8850 | train loss: 0.0135 | f1分数: 87.8501 | time: 494.3672
# Step: 8900 | train loss: 0.0036 | f1分数: 89.0772 | time: 497.2329
# Step: 8950 | train loss: 0.0113 | f1分数: 89.3134 | time: 500.0791
# Step: 9000 | train loss: 0.0019 | f1分数: 89.0965 | time: 502.9518
# Step: 9050 | train loss: 0.0023 | f1分数: 89.1292 | time: 505.8110
# Step: 9100 | train loss: 0.0162 | f1分数: 88.4172 | time: 508.6687
# Step: 9150 | train loss: 0.0259 | f1分数: 89.0353 | time: 511.4585
# Step: 9200 | train loss: 0.0038 | f1分数: 89.0996 | time: 514.2752
# Step: 9250 | train loss: 0.0033 | f1分数: 88.6138 | time: 517.1224
# Step: 9300 | train loss: 0.0093 | f1分数: 88.6936 | time: 519.9952
# Step: 9350 | train loss: 0.0056 | f1分数: 89.0343 | time: 522.8582
# Step: 9400 | train loss: 0.0011 | f1分数: 88.5738 | time: 525.6937
# Step: 9450 | train loss: 0.0411 | f1分数: 88.8932 | time: 528.5048
# Step: 9500 | train loss: 0.0008 | f1分数: 88.1171 | time: 531.3568
# Step: 9550 | train loss: 0.0120 | f1分数: 89.2259 | time: 534.2058
# Step: 9600 | train loss: 0.0207 | f1分数: 87.2480 | time: 537.0314
# Step: 9650 | train loss: 0.0115 | f1分数: 87.1735 | time: 539.8661
# Step: 9700 | train loss: 0.0423 | f1分数: 87.0006 | time: 542.6956
# Step: 9750 | train loss: 0.0084 | f1分数: 88.4903 | time: 545.5427
# Step: 9800 | train loss: 0.0170 | f1分数: 88.6627 | time: 548.3671
# Step: 9850 | train loss: 0.0015 | f1分数: 89.1064 | time: 551.1826
# Step: 9900 | train loss: 0.0011 | f1分数: 89.0492 | time: 554.0299
# Step: 9950 | train loss: 0.0051 | f1分数: 88.7337 | time: 556.8750
# Step: 10000 | train loss: 0.0053 | f1分数: 88.8186 | time: 559.7347
# Step: 10050 | train loss: 0.0340 | f1分数: 88.4558 | time: 562.5425
# Step: 10100 | train loss: 0.0049 | f1分数: 88.3894 | time: 565.3424
# Step: 10150 | train loss: 0.0054 | f1分数: 88.7508 | time: 568.2063
# Step: 10200 | train loss: 0.0042 | f1分数: 88.9498 | time: 571.0305
# Step: 10250 | train loss: 0.0198 | f1分数: 88.8365 | time: 573.8502
# Step: 10300 | train loss: 0.0011 | f1分数: 89.1219 | time: 576.6669
# Step: 10350 | train loss: 0.0602 | f1分数: 88.3982 | time: 579.5212
# Step: 10400 | train loss: 0.0062 | f1分数: 88.9183 | time: 582.3164
# Step: 10450 | train loss: 0.0083 | f1分数: 88.8787 | time: 585.1665
# Step: 10500 | train loss: 0.0041 | f1分数: 88.9612 | time: 588.0179
# Step: 10550 | train loss: 0.0006 | f1分数: 88.9511 | time: 590.8636
# Step: 10600 | train loss: 0.0082 | f1分数: 89.3567 | time: 593.7259
# Step: 10650 | train loss: 0.0096 | f1分数: 89.0643 | time: 596.5637
# Step: 10700 | train loss: 0.0016 | f1分数: 86.3017 | time: 599.4225
# Step: 10750 | train loss: 0.0076 | f1分数: 88.7583 | time: 602.2322
# Step: 10800 | train loss: 0.0220 | f1分数: 89.0494 | time: 605.0102
# Step: 10850 | train loss: 0.0265 | f1分数: 88.6791 | time: 607.7861
# Step: 10900 | train loss: 0.0035 | f1分数: 88.9940 | time: 610.5689
# Step: 10950 | train loss: 0.0046 | f1分数: 88.9728 | time: 613.3655
# Step: 11000 | train loss: 0.0085 | f1分数: 88.8454 | time: 616.2101
# Step: 11050 | train loss: 0.0025 | f1分数: 89.1167 | time: 619.0452
# Step: 11100 | train loss: 0.0614 | f1分数: 89.0755 | time: 621.8898
# Step: 11150 | train loss: 0.0216 | f1分数: 88.1307 | time: 624.7315
# Step: 11200 | train loss: 0.0254 | f1分数: 88.8034 | time: 627.5755
# Step: 11250 | train loss: 0.0034 | f1分数: 89.2334 | time: 630.4521
# Step: 11300 | train loss: 0.0033 | f1分数: 89.2347 | time: 633.2461
# Step: 11350 | train loss: 0.0002 | f1分数: 89.2203 | time: 636.1195
# Step: 11400 | train loss: 0.0006 | f1分数: 88.8232 | time: 638.9902
# Step: 11450 | train loss: 0.0003 | f1分数: 88.9742 | time: 641.8719
# Step: 11500 | train loss: 0.0062 | f1分数: 89.0522 | time: 644.7373
# Step: 11550 | train loss: 0.0013 | f1分数: 89.1155 | time: 647.5688
# Step: 11600 | train loss: 0.0014 | f1分数: 89.1019 | time: 650.3856
# Step: 11650 | train loss: 0.0005 | f1分数: 88.5304 | time: 653.2158
# Step: 11700 | train loss: 0.0198 | f1分数: 88.7325 | time: 656.0540
# Step: 11750 | train loss: 0.0080 | f1分数: 88.2169 | time: 658.9422
# Step: 11800 | train loss: 0.0146 | f1分数: 89.0457 | time: 661.7789
# Step: 11850 | train loss: 0.0068 | f1分数: 88.0627 | time: 664.6003
# Step: 11900 | train loss: 0.0121 | f1分数: 88.4536 | time: 667.4559
# Step: 11950 | train loss: 0.0097 | f1分数: 89.2306 | time: 670.3004
# Step: 12000 | train loss: 0.0138 | f1分数: 89.1228 | time: 673.1584
# Step: 12050 | train loss: 0.0015 | f1分数: 89.0739 | time: 675.9660
# Step: 12100 | train loss: 0.0329 | f1分数: 89.0142 | time: 678.8014
# Step: 12150 | train loss: 0.0012 | f1分数: 88.7189 | time: 681.6108
# Step: 12200 | train loss: 0.0090 | f1分数: 88.8671 | time: 684.3894
# Step: 12250 | train loss: 0.0048 | f1分数: 89.1079 | time: 687.1532
# Step: 12300 | train loss: 0.0008 | f1分数: 88.8668 | time: 689.9481
# Step: 12350 | train loss: 0.0019 | f1分数: 88.7823 | time: 692.7330
# Step: 12400 | train loss: 0.0002 | f1分数: 88.6557 | time: 695.5394
# Step: 12450 | train loss: 0.0018 | f1分数: 88.4862 | time: 698.3803
# Step: 12500 | train loss: 0.0017 | f1分数: 88.9829 | time: 701.1819
# Step: 12550 | train loss: 0.0009 | f1分数: 89.3258 | time: 704.0337
# Step: 12600 | train loss: 0.0043 | f1分数: 88.4433 | time: 706.8598
# Step: 12650 | train loss: 0.0261 | f1分数: 88.9634 | time: 709.7159
# Step: 12700 | train loss: 0.0026 | f1分数: 88.9794 | time: 712.5395
# Step: 12750 | train loss: 0.0049 | f1分数: 88.7775 | time: 715.4328
# Step: 12800 | train loss: 0.0027 | f1分数: 88.0901 | time: 718.2590
# Step: 12850 | train loss: 0.0062 | f1分数: 88.5767 | time: 721.1203
# Step: 12900 | train loss: 0.0333 | f1分数: 88.8426 | time: 723.9723
# Step: 12950 | train loss: 0.0052 | f1分数: 88.9894 | time: 726.8325
# Step: 13000 | train loss: 0.0823 | f1分数: 89.3716 | time: 729.6758
# Step: 13050 | train loss: 0.0315 | f1分数: 88.0588 | time: 732.5459
# Step: 13100 | train loss: 0.0390 | f1分数: 88.4475 | time: 735.3834
# Step: 13150 | train loss: 0.0085 | f1分数: 88.6062 | time: 738.2265
# Step: 13200 | train loss: 0.0204 | f1分数: 88.8256 | time: 741.0600
# Step: 13250 | train loss: 0.0103 | f1分数: 88.6474 | time: 743.8616
# Step: 13300 | train loss: 0.0255 | f1分数: 88.7038 | time: 746.6611
# Step: 13350 | train loss: 0.0036 | f1分数: 89.5744 | time: 749.4910
# Step: 13400 | train loss: 0.0018 | f1分数: 88.8624 | time: 752.3122
# Step: 13450 | train loss: 0.0137 | f1分数: 89.1297 | time: 755.1528
# Step: 13500 | train loss: 0.0116 | f1分数: 88.8809 | time: 758.0060
# Step: 13550 | train loss: 0.0016 | f1分数: 89.1378 | time: 760.8411
# Step: 13600 | train loss: 0.0264 | f1分数: 88.8091 | time: 763.6971
# Step: 13650 | train loss: 0.0088 | f1分数: 89.4945 | time: 766.5547
# Step: 13700 | train loss: 0.0005 | f1分数: 89.3138 | time: 769.3943
# Step: 13750 | train loss: 0.0024 | f1分数: 89.4933 | time: 772.1907
# Step: 13800 | train loss: 0.0009 | f1分数: 89.6853 | time: 775.0100
# Step: 13850 | train loss: 0.0014 | f1分数: 89.4359 | time: 777.8506
# Step: 13900 | train loss: 0.0002 | f1分数: 89.5710 | time: 780.7206

# batch_norm
# Step: 50 | train loss: 1.8847 | f1分数: 75.4185 | time: 8.5451
# Step: 100 | train loss: 0.5524 | f1分数: 73.2357 | time: 15.5065
# Step: 150 | train loss: 0.5216 | f1分数: 79.4564 | time: 22.4936
# Step: 200 | train loss: 0.4163 | f1分数: 81.7525 | time: 29.4083
# Step: 250 | train loss: 0.4865 | f1分数: 82.4708 | time: 36.3883
# Step: 300 | train loss: 0.8879 | f1分数: 82.4447 | time: 43.3647
# Step: 350 | train loss: 0.1998 | f1分数: 79.1075 | time: 50.3042
# Step: 400 | train loss: 0.6005 | f1分数: 84.5346 | time: 57.3018
# Step: 450 | train loss: 0.3947 | f1分数: 86.7070 | time: 64.1632
# Step: 500 | train loss: 0.4155 | f1分数: 87.2594 | time: 71.0157
# Step: 550 | train loss: 0.4372 | f1分数: 86.1199 | time: 77.8393
# Step: 600 | train loss: 0.1684 | f1分数: 86.3683 | time: 84.6702
# Step: 650 | train loss: 0.2898 | f1分数: 85.6111 | time: 91.4920
# Step: 700 | train loss: 0.4082 | f1分数: 87.8272 | time: 98.3276
# Step: 750 | train loss: 0.2913 | f1分数: 86.7265 | time: 105.1262
# Step: 800 | train loss: 0.2579 | f1分数: 86.7274 | time: 111.9468
# Step: 850 | train loss: 0.2578 | f1分数: 84.5074 | time: 118.7825
# Step: 900 | train loss: 0.4121 | f1分数: 82.2960 | time: 125.6352
# Step: 950 | train loss: 0.2924 | f1分数: 85.1274 | time: 132.4797
# Step: 1000 | train loss: 0.2183 | f1分数: 86.2343 | time: 139.3060
# Step: 1050 | train loss: 0.2254 | f1分数: 86.3942 | time: 146.1484
# Step: 1100 | train loss: 0.1235 | f1分数: 87.3275 | time: 153.0189
# Step: 1150 | train loss: 0.3089 | f1分数: 87.3488 | time: 159.8399
# Step: 1200 | train loss: 0.1525 | f1分数: 86.7554 | time: 166.6699
# Step: 1250 | train loss: 0.1697 | f1分数: 84.3454 | time: 173.4861
# Step: 1300 | train loss: 0.1461 | f1分数: 87.9694 | time: 180.3358
# Step: 1350 | train loss: 0.2737 | f1分数: 81.5340 | time: 187.1729
# Step: 1400 | train loss: 0.0520 | f1分数: 88.1009 | time: 193.9845
# Step: 1450 | train loss: 0.1037 | f1分数: 88.8072 | time: 200.8020
# Step: 1500 | train loss: 0.0745 | f1分数: 86.1080 | time: 207.6485
# Step: 1550 | train loss: 0.0576 | f1分数: 89.2631 | time: 214.4767
# Step: 1600 | train loss: 0.0785 | f1分数: 88.7744 | time: 221.3371
# Step: 1650 | train loss: 0.1163 | f1分数: 88.0037 | time: 228.1430
# Step: 1700 | train loss: 0.0489 | f1分数: 89.3755 | time: 234.9848
# Step: 1750 | train loss: 0.2123 | f1分数: 88.9874 | time: 241.8300
# Step: 1800 | train loss: 0.0955 | f1分数: 89.0948 | time: 248.6194
# Step: 1850 | train loss: 0.1211 | f1分数: 88.2939 | time: 255.4526
# Step: 1900 | train loss: 0.0509 | f1分数: 90.2670 | time: 262.3110
# Step: 1950 | train loss: 0.0587 | f1分数: 89.9386 | time: 269.1338
# Step: 2000 | train loss: 0.1092 | f1分数: 86.5734 | time: 275.9886
# Step: 2050 | train loss: 0.0765 | f1分数: 90.6335 | time: 282.8138
# Step: 2100 | train loss: 0.0598 | f1分数: 89.2628 | time: 289.6598
# Step: 2150 | train loss: 0.0911 | f1分数: 88.9488 | time: 296.4865
# Step: 2200 | train loss: 0.0403 | f1分数: 89.8280 | time: 303.2844
# Step: 2250 | train loss: 0.0601 | f1分数: 89.1585 | time: 310.1086
# Step: 2300 | train loss: 0.1033 | f1分数: 90.4390 | time: 316.9799
# Step: 2350 | train loss: 0.0447 | f1分数: 90.4294 | time: 323.8123
# Step: 2400 | train loss: 0.0430 | f1分数: 90.2959 | time: 330.6512
# Step: 2450 | train loss: 0.0337 | f1分数: 89.3536 | time: 337.4828
# Step: 2500 | train loss: 0.0846 | f1分数: 90.2394 | time: 344.3554
# Step: 2550 | train loss: 0.0369 | f1分数: 83.2276 | time: 351.1841
# Step: 2600 | train loss: 0.0240 | f1分数: 88.1938 | time: 358.0474
# Step: 2650 | train loss: 0.0537 | f1分数: 89.2393 | time: 364.8911
# Step: 2700 | train loss: 0.0585 | f1分数: 86.6932 | time: 371.7469
# Step: 2750 | train loss: 0.0168 | f1分数: 89.6535 | time: 378.5966
# Step: 2800 | train loss: 0.0561 | f1分数: 89.8143 | time: 385.4494
# Step: 2850 | train loss: 0.0156 | f1分数: 88.8576 | time: 392.3150
# Step: 2900 | train loss: 0.0281 | f1分数: 89.4107 | time: 399.1333
# Step: 2950 | train loss: 0.1003 | f1分数: 88.1408 | time: 405.9618
# Step: 3000 | train loss: 0.0115 | f1分数: 88.1397 | time: 412.8293
# Step: 3050 | train loss: 0.1154 | f1分数: 88.0743 | time: 419.6276
# Step: 3100 | train loss: 0.0486 | f1分数: 89.2958 | time: 426.4608
# Step: 3150 | train loss: 0.0396 | f1分数: 89.4840 | time: 433.3275
# Step: 3200 | train loss: 0.0342 | f1分数: 89.0755 | time: 440.1930
# Step: 3250 | train loss: 0.0394 | f1分数: 89.2509 | time: 446.9923
# Step: 3300 | train loss: 0.0082 | f1分数: 88.6139 | time: 453.8248
# Step: 3350 | train loss: 0.0529 | f1分数: 89.4834 | time: 460.7073
# Step: 3400 | train loss: 0.0329 | f1分数: 90.3767 | time: 467.5589
# Step: 3450 | train loss: 0.0388 | f1分数: 89.3641 | time: 474.3973
# Step: 3500 | train loss: 0.0281 | f1分数: 88.7649 | time: 481.2377
# Step: 3550 | train loss: 0.0609 | f1分数: 89.3677 | time: 488.0852
# Step: 3600 | train loss: 0.0143 | f1分数: 89.8128 | time: 494.9244
# Step: 3650 | train loss: 0.0256 | f1分数: 90.1847 | time: 501.7637
# Step: 3700 | train loss: 0.0079 | f1分数: 89.7448 | time: 508.6198
# Step: 3750 | train loss: 0.0115 | f1分数: 90.0800 | time: 515.4630
# Step: 3800 | train loss: 0.0681 | f1分数: 89.8229 | time: 522.3133
# Step: 3850 | train loss: 0.0337 | f1分数: 91.1091 | time: 529.1354
# Step: 3900 | train loss: 0.1090 | f1分数: 83.1708 | time: 535.9951
# Step: 3950 | train loss: 0.0216 | f1分数: 89.7002 | time: 542.8872
# Step: 4000 | train loss: 0.0414 | f1分数: 89.8086 | time: 549.7377
# Step: 4050 | train loss: 0.0104 | f1分数: 90.2200 | time: 556.6190
# Step: 4100 | train loss: 0.0022 | f1分数: 90.1670 | time: 563.4582
# Step: 4150 | train loss: 0.0590 | f1分数: 89.5631 | time: 570.2904
# Step: 4200 | train loss: 0.0257 | f1分数: 89.8108 | time: 577.1241
# Step: 4250 | train loss: 0.0366 | f1分数: 88.7267 | time: 583.9781
# Step: 4300 | train loss: 0.0363 | f1分数: 89.7701 | time: 590.7809
# Step: 4350 | train loss: 0.0045 | f1分数: 89.6890 | time: 597.6101
# Step: 4400 | train loss: 0.0142 | f1分数: 88.6845 | time: 604.4450
# Step: 4450 | train loss: 0.0034 | f1分数: 90.1715 | time: 611.3038
# Step: 4500 | train loss: 0.0189 | f1分数: 90.4580 | time: 618.1574
# Step: 4550 | train loss: 0.0017 | f1分数: 90.5438 | time: 624.9766
# Step: 4600 | train loss: 0.0051 | f1分数: 90.1347 | time: 631.7817
# Step: 4650 | train loss: 0.0122 | f1分数: 90.0890 | time: 638.6324
# Step: 4700 | train loss: 0.0264 | f1分数: 90.1505 | time: 645.4688
# Step: 4750 | train loss: 0.0053 | f1分数: 89.5256 | time: 652.3045
# Step: 4800 | train loss: 0.0353 | f1分数: 90.1112 | time: 659.1550
# Step: 4850 | train loss: 0.0013 | f1分数: 90.6499 | time: 666.0003
# Step: 4900 | train loss: 0.0107 | f1分数: 89.2050 | time: 672.8869
# Step: 4950 | train loss: 0.0213 | f1分数: 88.4746 | time: 679.7255
# Step: 5000 | train loss: 0.0036 | f1分数: 90.3788 | time: 686.6094
# Step: 5050 | train loss: 0.0102 | f1分数: 89.6364 | time: 693.4561
# Step: 5100 | train loss: 0.0847 | f1分数: 88.1539 | time: 700.3141
# Step: 5150 | train loss: 0.0469 | f1分数: 87.6588 | time: 707.1794
# Step: 5200 | train loss: 0.0211 | f1分数: 88.7409 | time: 714.0592
# Step: 5250 | train loss: 0.0033 | f1分数: 86.0808 | time: 720.9224
# Step: 5300 | train loss: 0.0549 | f1分数: 89.8179 | time: 727.7393
# Step: 5350 | train loss: 0.0169 | f1分数: 89.1414 | time: 734.5849
# Step: 5400 | train loss: 0.0517 | f1分数: 90.5571 | time: 741.4411
# Step: 5450 | train loss: 0.0115 | f1分数: 87.6478 | time: 748.2728
# Step: 5500 | train loss: 0.0194 | f1分数: 89.2416 | time: 755.1413
# Step: 5550 | train loss: 0.0469 | f1分数: 89.7177 | time: 761.9887
# Step: 5600 | train loss: 0.0523 | f1分数: 90.1847 | time: 768.8553
# Step: 5650 | train loss: 0.0101 | f1分数: 89.6734 | time: 775.7321
# Step: 5700 | train loss: 0.0963 | f1分数: 89.3828 | time: 782.5591
# Step: 5750 | train loss: 0.0060 | f1分数: 89.8066 | time: 789.4218
# Step: 5800 | train loss: 0.0175 | f1分数: 90.4341 | time: 796.2717
# Step: 5850 | train loss: 0.0172 | f1分数: 89.9180 | time: 803.1481
# Step: 5900 | train loss: 0.0068 | f1分数: 90.1920 | time: 809.9771
# Step: 5950 | train loss: 0.0028 | f1分数: 90.4523 | time: 816.8224
# Step: 6000 | train loss: 0.0107 | f1分数: 90.4286 | time: 823.6707
# Step: 6050 | train loss: 0.0175 | f1分数: 90.6740 | time: 830.4833
# Step: 6100 | train loss: 0.0125 | f1分数: 90.9264 | time: 837.3079
# Step: 6150 | train loss: 0.0028 | f1分数: 90.5883 | time: 844.1288
# Step: 6200 | train loss: 0.0055 | f1分数: 90.1005 | time: 850.9537
# Step: 6250 | train loss: 0.0003 | f1分数: 90.2043 | time: 857.7930
# Step: 6300 | train loss: 0.0102 | f1分数: 88.6565 | time: 864.6621
# Step: 6350 | train loss: 0.0075 | f1分数: 88.3215 | time: 871.4823
# Step: 6400 | train loss: 0.0106 | f1分数: 89.9149 | time: 878.3254
# Step: 6450 | train loss: 0.0237 | f1分数: 89.5158 | time: 885.1392
# Step: 6500 | train loss: 0.0184 | f1分数: 91.0294 | time: 891.9868
# Step: 6550 | train loss: 0.0375 | f1分数: 89.2992 | time: 898.7692
# Step: 6600 | train loss: 0.0181 | f1分数: 89.8174 | time: 905.6249
# Step: 6650 | train loss: 0.0214 | f1分数: 89.9797 | time: 912.4679
# Step: 6700 | train loss: 0.0028 | f1分数: 91.1261 | time: 919.3250
# Step: 6750 | train loss: 0.0009 | f1分数: 90.6133 | time: 926.1822
# Step: 6800 | train loss: 0.0303 | f1分数: 90.5259 | time: 933.0211
# Step: 6850 | train loss: 0.0003 | f1分数: 90.6507 | time: 939.8843
# Step: 6900 | train loss: 0.0054 | f1分数: 90.9088 | time: 946.7258
# Step: 6950 | train loss: 0.0030 | f1分数: 90.7539 | time: 953.5696
# Step: 7000 | train loss: 0.0003 | f1分数: 90.2467 | time: 960.4011
# Step: 7050 | train loss: 0.0015 | f1分数: 90.6329 | time: 967.2624
# Step: 7100 | train loss: 0.0012 | f1分数: 89.9727 | time: 974.1161
# Step: 7150 | train loss: 0.0056 | f1分数: 90.7375 | time: 980.9485
# Step: 7200 | train loss: 0.0014 | f1分数: 89.1030 | time: 987.8208
# Step: 7250 | train loss: 0.0015 | f1分数: 90.6395 | time: 994.6634
# Step: 7300 | train loss: 0.0126 | f1分数: 82.4758 | time: 1001.5181
# Step: 7350 | train loss: 0.0143 | f1分数: 88.6999 | time: 1008.3644
# Step: 7400 | train loss: 0.0196 | f1分数: 88.8031 | time: 1015.2444
# Step: 7450 | train loss: 0.0220 | f1分数: 89.2483 | time: 1022.0765
# Step: 7500 | train loss: 0.2442 | f1分数: 88.4045 | time: 1028.9654
# Step: 7550 | train loss: 0.0043 | f1分数: 88.7442 | time: 1035.8197
# Step: 7600 | train loss: 0.0384 | f1分数: 89.5643 | time: 1042.6404
# Step: 7650 | train loss: 0.0036 | f1分数: 89.8144 | time: 1049.4765
# Step: 7700 | train loss: 0.0099 | f1分数: 90.2991 | time: 1056.3066
# Step: 7750 | train loss: 0.0109 | f1分数: 90.2558 | time: 1063.1671
# Step: 7800 | train loss: 0.0097 | f1分数: 90.8159 | time: 1070.0112
# Step: 7850 | train loss: 0.0036 | f1分数: 90.2218 | time: 1076.8933
# Step: 7900 | train loss: 0.0006 | f1分数: 90.6870 | time: 1083.7216
# Step: 7950 | train loss: 0.0043 | f1分数: 90.6602 | time: 1090.5719
# Step: 8000 | train loss: 0.0016 | f1分数: 90.8603 | time: 1097.4191
# Step: 8050 | train loss: 0.0188 | f1分数: 89.9631 | time: 1104.2714
# Step: 8100 | train loss: 0.0003 | f1分数: 90.8453 | time: 1111.1369
# Step: 8150 | train loss: 0.0236 | f1分数: 90.8354 | time: 1117.9984
# Step: 8200 | train loss: 0.0094 | f1分数: 89.9771 | time: 1124.8508
# Step: 8250 | train loss: 0.0014 | f1分数: 90.8378 | time: 1131.6571
# Step: 8300 | train loss: 0.0061 | f1分数: 90.5388 | time: 1138.5082
# Step: 8350 | train loss: 0.0101 | f1分数: 89.7075 | time: 1145.3652
# Step: 8400 | train loss: 0.0002 | f1分数: 90.8635 | time: 1152.1941
# Step: 8450 | train loss: 0.0428 | f1分数: 88.5023 | time: 1159.0424
# Step: 8500 | train loss: 0.0245 | f1分数: 89.3052 | time: 1165.8997
# Step: 8550 | train loss: 0.0417 | f1分数: 89.8052 | time: 1172.7833
# Step: 8600 | train loss: 0.0236 | f1分数: 90.1511 | time: 1179.6395
# Step: 8650 | train loss: 0.0255 | f1分数: 89.5535 | time: 1186.4665
# Step: 8700 | train loss: 0.0001 | f1分数: 89.1902 | time: 1193.2911
# Step: 8750 | train loss: 0.0095 | f1分数: 89.3006 | time: 1200.1413
# Step: 8800 | train loss: 0.0083 | f1分数: 89.3069 | time: 1206.9773
# Step: 8850 | train loss: 0.0223 | f1分数: 89.1098 | time: 1213.8296
# Step: 8900 | train loss: 0.0042 | f1分数: 90.2959 | time: 1220.6523
# Step: 8950 | train loss: 0.0195 | f1分数: 90.5992 | time: 1227.5340
# Step: 9000 | train loss: 0.0059 | f1分数: 89.9698 | time: 1234.3813
# Step: 9050 | train loss: 0.0035 | f1分数: 90.8559 | time: 1241.2292
# Step: 9100 | train loss: 0.0040 | f1分数: 90.9393 | time: 1248.0739
# Step: 9150 | train loss: 0.0044 | f1分数: 90.9447 | time: 1254.9523
# Step: 9200 | train loss: 0.0004 | f1分数: 90.9369 | time: 1261.7811
# Step: 9250 | train loss: 0.0012 | f1分数: 90.8139 | time: 1268.6037
# Step: 9300 | train loss: 0.0004 | f1分数: 90.6976 | time: 1275.4940
# Step: 9350 | train loss: 0.0002 | f1分数: 90.7101 | time: 1282.3806
# Step: 9400 | train loss: 0.0000 | f1分数: 90.6977 | time: 1289.2257
# Step: 9450 | train loss: 0.0017 | f1分数: 90.6866 | time: 1296.0544
# Step: 9500 | train loss: 0.0000 | f1分数: 90.6782 | time: 1302.9368
# Step: 9550 | train loss: 0.0003 | f1分数: 90.5941 | time: 1309.7889
# Step: 9600 | train loss: 0.0002 | f1分数: 90.6101 | time: 1316.6462
# Step: 9650 | train loss: 0.0001 | f1分数: 90.6724 | time: 1323.5011
# Step: 9700 | train loss: 0.0001 | f1分数: 90.8164 | time: 1330.3459
# Step: 9750 | train loss: 0.0000 | f1分数: 90.7828 | time: 1337.2078
# Step: 9800 | train loss: 0.0002 | f1分数: 90.5717 | time: 1344.0773
# Step: 9850 | train loss: 0.0002 | f1分数: 90.8067 | time: 1350.9315
# Step: 9900 | train loss: 0.0001 | f1分数: 90.6368 | time: 1357.8041
# Step: 9950 | train loss: 0.0006 | f1分数: 90.7225 | time: 1364.6598
# Step: 10000 | train loss: 0.0001 | f1分数: 90.7056 | time: 1371.5333
# Step: 10050 | train loss: 0.0002 | f1分数: 90.6802 | time: 1378.3731
# Step: 10100 | train loss: 0.0001 | f1分数: 90.7992 | time: 1385.2390
# Step: 10150 | train loss: 0.0004 | f1分数: 90.3063 | time: 1392.0807
# Step: 10200 | train loss: 0.0059 | f1分数: 89.8681 | time: 1398.9416
# Step: 10250 | train loss: 0.0007 | f1分数: 90.5128 | time: 1405.7934
# Step: 10300 | train loss: 0.0002 | f1分数: 90.5283 | time: 1412.6152
# Step: 10350 | train loss: 0.0150 | f1分数: 88.6214 | time: 1419.5061
# Step: 10400 | train loss: 0.0078 | f1分数: 88.6891 | time: 1426.3489
# Step: 10450 | train loss: 0.0670 | f1分数: 89.5043 | time: 1433.1853
# Step: 10500 | train loss: 0.0457 | f1分数: 89.0571 | time: 1439.9898
# Step: 10550 | train loss: 0.0139 | f1分数: 86.0685 | time: 1446.8451
# Step: 10600 | train loss: 0.0120 | f1分数: 90.1329 | time: 1453.7021
# Step: 10650 | train loss: 0.0211 | f1分数: 90.7684 | time: 1460.5665
# Step: 10700 | train loss: 0.0206 | f1分数: 89.2011 | time: 1467.4144
# Step: 10750 | train loss: 0.0659 | f1分数: 90.1098 | time: 1474.2882
# Step: 10800 | train loss: 0.0158 | f1分数: 90.3367 | time: 1481.1453
# Step: 10850 | train loss: 0.0106 | f1分数: 89.2327 | time: 1487.9479
# Step: 10900 | train loss: 0.0242 | f1分数: 90.6761 | time: 1494.7994
# Step: 10950 | train loss: 0.0139 | f1分数: 90.4429 | time: 1501.6561
# Step: 11000 | train loss: 0.0095 | f1分数: 90.4951 | time: 1508.4408
# Step: 11050 | train loss: 0.0007 | f1分数: 89.8908 | time: 1515.2786
# Step: 11100 | train loss: 0.0081 | f1分数: 90.1183 | time: 1522.1498
# Step: 11150 | train loss: 0.0082 | f1分数: 90.6104 | time: 1528.9730
# Step: 11200 | train loss: 0.0009 | f1分数: 90.2874 | time: 1535.8226
# Step: 11250 | train loss: 0.0002 | f1分数: 90.9042 | time: 1542.6783
# Step: 11300 | train loss: 0.0003 | f1分数: 90.6687 | time: 1549.5494
# Step: 11350 | train loss: 0.0001 | f1分数: 90.8957 | time: 1556.3946
# Step: 11400 | train loss: 0.0000 | f1分数: 90.9051 | time: 1563.2047
# Step: 11450 | train loss: 0.0001 | f1分数: 90.9409 | time: 1570.0266
# Step: 11500 | train loss: 0.0010 | f1分数: 90.8356 | time: 1576.8791
# Step: 11550 | train loss: 0.0001 | f1分数: 90.7895 | time: 1583.7517
# Step: 11600 | train loss: 0.0002 | f1分数: 90.6765 | time: 1590.6058
# Step: 11650 | train loss: 0.0000 | f1分数: 90.6516 | time: 1597.4741
# Step: 11700 | train loss: 0.0002 | f1分数: 90.7570 | time: 1604.3384
# Step: 11750 | train loss: 0.0000 | f1分数: 90.7835 | time: 1611.1794
# Step: 11800 | train loss: 0.0002 | f1分数: 90.9372 | time: 1618.0368
# Step: 11850 | train loss: 0.0000 | f1分数: 90.5881 | time: 1624.8972
# Step: 11900 | train loss: 0.0004 | f1分数: 90.8266 | time: 1631.7488
# Step: 11950 | train loss: 0.0001 | f1分数: 90.6618 | time: 1638.5984
# Step: 12000 | train loss: 0.0001 | f1分数: 90.7951 | time: 1645.3975
# Step: 12050 | train loss: 0.0000 | f1分数: 90.4380 | time: 1652.2581
# Step: 12100 | train loss: 0.0071 | f1分数: 90.7183 | time: 1659.1425
# Step: 12150 | train loss: 0.0000 | f1分数: 90.7810 | time: 1665.9774
# Step: 12200 | train loss: 0.0006 | f1分数: 89.1734 | time: 1672.8039
# Step: 12250 | train loss: 0.0078 | f1分数: 89.1652 | time: 1679.6780
# Step: 12300 | train loss: 0.0014 | f1分数: 88.7763 | time: 1686.5083
# Step: 12350 | train loss: 0.0242 | f1分数: 89.3335 | time: 1693.3256
# Step: 12400 | train loss: 0.0029 | f1分数: 89.7945 | time: 1700.1494
# Step: 12450 | train loss: 0.0227 | f1分数: 89.2218 | time: 1706.9835
# Step: 12500 | train loss: 0.0150 | f1分数: 90.2233 | time: 1713.8137
# Step: 12550 | train loss: 0.0029 | f1分数: 88.2457 | time: 1720.6844
# Step: 12600 | train loss: 0.0439 | f1分数: 89.5772 | time: 1727.5236
# Step: 12650 | train loss: 0.0195 | f1分数: 89.9977 | time: 1734.3757
# Step: 12700 | train loss: 0.0245 | f1分数: 91.1260 | time: 1741.2055
# Step: 12750 | train loss: 0.0106 | f1分数: 90.9327 | time: 1748.0757
# Step: 12800 | train loss: 0.0004 | f1分数: 90.8243 | time: 1754.8909
# Step: 12850 | train loss: 0.0092 | f1分数: 90.9239 | time: 1761.7442
# Step: 12900 | train loss: 0.0241 | f1分数: 90.1030 | time: 1768.6291
# Step: 12950 | train loss: 0.0006 | f1分数: 90.9201 | time: 1775.4722
# Step: 13000 | train loss: 0.0087 | f1分数: 90.5658 | time: 1782.3064
# Step: 13050 | train loss: 0.0040 | f1分数: 90.8190 | time: 1789.1273
# Step: 13100 | train loss: 0.0006 | f1分数: 91.2555 | time: 1795.9578
# Step: 13150 | train loss: 0.0005 | f1分数: 91.3064 | time: 1802.7992
# Step: 13200 | train loss: 0.0001 | f1分数: 91.2655 | time: 1809.6533
# Step: 13250 | train loss: 0.0001 | f1分数: 91.3412 | time: 1816.4944
# Step: 13300 | train loss: 0.0002 | f1分数: 91.2041 | time: 1823.3613
# Step: 13350 | train loss: 0.0002 | f1分数: 91.2215 | time: 1830.1991
# Step: 13400 | train loss: 0.0001 | f1分数: 91.1247 | time: 1837.0508
# Step: 13450 | train loss: 0.0234 | f1分数: 91.0983 | time: 1843.8870
# Step: 13500 | train loss: 0.0026 | f1分数: 91.0950 | time: 1850.7464
# Step: 13550 | train loss: 0.0000 | f1分数: 91.0807 | time: 1857.5803
# Step: 13600 | train loss: 0.0001 | f1分数: 91.1387 | time: 1864.4403
# Step: 13650 | train loss: 0.0001 | f1分数: 91.0313 | time: 1871.2873
# Step: 13700 | train loss: 0.0001 | f1分数: 91.0758 | time: 1878.1624
# Step: 13750 | train loss: 0.0005 | f1分数: 91.0457 | time: 1885.0116
# Step: 13800 | train loss: 0.0018 | f1分数: 90.8840 | time: 1891.8390
# Step: 13850 | train loss: 0.0045 | f1分数: 91.3751 | time: 1898.7015
# Step: 13900 | train loss: 0.0002 | f1分数: 91.4542 | time: 1905.5453

# Step: 50 | train loss: 2.0156 | f1分数: 75.4185 | time: 7.7555
# 精度: 69.0425%
# 召回率: 83.0918%
# Step: 100 | train loss: 0.4887 | f1分数: 76.4352 | time: 13.4591
# 精度: 71.8136%
# 召回率: 81.8367%
# Step: 150 | train loss: 0.5033 | f1分数: 79.3477 | time: 19.2229
# 精度: 77.4482%
# 召回率: 83.8878%
# Step: 200 | train loss: 0.3983 | f1分数: 82.2664 | time: 24.9973
# 精度: 81.5298%
# 召回率: 84.2347%
# Step: 250 | train loss: 0.4814 | f1分数: 82.3311 | time: 30.7822
# 精度: 82.6849%
# 召回率: 84.8878%
# Step: 300 | train loss: 0.8629 | f1分数: 84.3402 | time: 36.5666
# 精度: 85.8475%
# 召回率: 85.2245%
# Step: 350 | train loss: 0.1712 | f1分数: 86.5176 | time: 42.3483
# 精度: 87.1363%
# 召回率: 86.6327%
# Step: 400 | train loss: 1.0013 | f1分数: 86.2086 | time: 48.1484
# 精度: 88.1864%
# 召回率: 85.6429%
# Step: 450 | train loss: 0.3237 | f1分数: 86.8776 | time: 53.9731
# 精度: 87.5093%
# 召回率: 87.4592%
# Step: 500 | train loss: 0.4938 | f1分数: 87.1539 | time: 59.8013
# 精度: 88.2162%
# 召回率: 87.7143%
# Step: 550 | train loss: 0.4001 | f1分数: 87.5809 | time: 65.6135
# 精度: 88.9562%
# 召回率: 87.8163%
# Step: 600 | train loss: 0.1836 | f1分数: 87.0473 | time: 71.4315
# 精度: 87.5536%
# 召回率: 88.5000%
# Step: 650 | train loss: 0.2042 | f1分数: 87.7064 | time: 77.2723
# 精度: 88.1382%
# 召回率: 88.9898%
# Step: 700 | train loss: 0.3349 | f1分数: 88.1288 | time: 83.1045
# 精度: 88.4145%
# 召回率: 89.3673%
# Step: 750 | train loss: 0.1884 | f1分数: 87.4073 | time: 88.9308
# 精度: 87.9214%
# 召回率: 88.4286%
# Step: 800 | train loss: 0.1968 | f1分数: 84.8704 | time: 94.7807
# 精度: 86.0239%
# 召回率: 87.8061%
# Step: 850 | train loss: 0.2525 | f1分数: 83.7985 | time: 100.6120
# 精度: 86.2765%
# 召回率: 86.7245%
# Step: 900 | train loss: 0.2109 | f1分数: 84.4798 | time: 106.4465
# 精度: 86.0832%
# 召回率: 86.7551%
# Step: 950 | train loss: 0.2058 | f1分数: 85.4742 | time: 112.3177
# 精度: 86.6358%
# 召回率: 86.3469%
# Step: 1000 | train loss: 0.1830 | f1分数: 86.0485 | time: 118.1686
# 精度: 87.5197%
# 召回率: 87.1939%
# Step: 1050 | train loss: 0.4938 | f1分数: 86.8133 | time: 124.0118
# 精度: 87.7969%
# 召回率: 87.4898%
# Step: 1100 | train loss: 0.1684 | f1分数: 86.6199 | time: 129.8274
# 精度: 87.0449%
# 召回率: 87.0102%
# Step: 1150 | train loss: 0.2935 | f1分数: 86.2147 | time: 135.6337
# 精度: 88.1185%
# 召回率: 85.5000%
# Step: 1200 | train loss: 0.1508 | f1分数: 86.5048 | time: 141.4795
# 精度: 87.2808%
# 召回率: 86.5306%
# Step: 1250 | train loss: 0.2084 | f1分数: 88.7899 | time: 147.3469
# 精度: 89.5493%
# 召回率: 89.0714%
# Step: 1300 | train loss: 0.1750 | f1分数: 88.5075 | time: 153.1641
# 精度: 89.8681%
# 召回率: 88.8367%
# Step: 1350 | train loss: 0.2323 | f1分数: 89.5877 | time: 159.0048
# 精度: 89.9915%
# 召回率: 90.0408%
# Step: 1400 | train loss: 0.0340 | f1分数: 87.8312 | time: 164.8201
# 精度: 89.7490%
# 召回率: 87.3571%
# Step: 1450 | train loss: 0.1188 | f1分数: 84.2902 | time: 170.6655
# 精度: 86.3109%
# 召回率: 83.4592%
# Step: 1500 | train loss: 0.0994 | f1分数: 87.5778 | time: 176.4735
# 精度: 88.1058%
# 召回率: 88.6020%
# Step: 1550 | train loss: 0.0596 | f1分数: 87.6930 | time: 182.3366
# 精度: 88.5122%
# 召回率: 88.4694%
# Step: 1600 | train loss: 0.0692 | f1分数: 89.2678 | time: 188.1817
# 精度: 89.7665%
# 召回率: 90.0918%
# Step: 1650 | train loss: 0.1051 | f1分数: 88.7789 | time: 194.0418
# 精度: 89.4545%
# 召回率: 89.0102%
# Step: 1700 | train loss: 0.0688 | f1分数: 85.3441 | time: 199.9170
# 精度: 89.1717%
# 召回率: 83.8163%
# Step: 1750 | train loss: 0.1281 | f1分数: 87.8769 | time: 205.8291
# 精度: 89.5393%
# 召回率: 87.7143%
# Step: 1800 | train loss: 0.0880 | f1分数: 86.4588 | time: 211.7645
# 精度: 88.9634%
# 召回率: 86.0306%
# Step: 1850 | train loss: 0.1056 | f1分数: 89.0903 | time: 217.7089
# 精度: 89.3791%
# 召回率: 89.9286%
# Step: 1900 | train loss: 0.0559 | f1分数: 82.1027 | time: 223.6453
# 精度: 87.0553%
# 召回率: 80.0612%
# Step: 1950 | train loss: 0.0925 | f1分数: 88.5709 | time: 229.5915
# 精度: 89.2440%
# 召回率: 89.5714%
# Step: 2000 | train loss: 0.0929 | f1分数: 90.2217 | time: 235.5087
# 精度: 90.7555%
# 召回率: 91.0306%
# Step: 2050 | train loss: 0.0895 | f1分数: 88.5072 | time: 241.4445
# 精度: 89.7435%
# 召回率: 89.9592%
# Step: 2100 | train loss: 0.0639 | f1分数: 82.2013 | time: 247.3545
# 精度: 85.4584%
# 召回率: 82.2143%
# Step: 2150 | train loss: 0.0772 | f1分数: 86.4165 | time: 253.2557
# 精度: 88.1669%
# 召回率: 87.7347%
# Step: 2200 | train loss: 0.0327 | f1分数: 87.8582 | time: 259.1842
# 精度: 88.6137%
# 召回率: 89.4592%
# Step: 2250 | train loss: 0.0349 | f1分数: 88.2300 | time: 265.1177
# 精度: 89.0080%
# 召回率: 89.7347%
# Step: 2300 | train loss: 0.0647 | f1分数: 89.2906 | time: 271.0570
# 精度: 89.5838%
# 召回率: 90.1429%
# Step: 2350 | train loss: 0.0379 | f1分数: 88.9473 | time: 276.9834
# 精度: 89.6710%
# 召回率: 90.0102%
# Step: 2400 | train loss: 0.0286 | f1分数: 89.0062 | time: 282.9331
# 精度: 89.5160%
# 召回率: 89.8673%
# Step: 2450 | train loss: 0.0142 | f1分数: 89.7349 | time: 288.8749
# 精度: 90.4592%
# 召回率: 90.0102%
# Step: 2500 | train loss: 0.0886 | f1分数: 88.9789 | time: 294.8084
# 精度: 89.5247%
# 召回率: 89.4490%
# Step: 2550 | train loss: 0.0457 | f1分数: 85.5011 | time: 300.7103
# 精度: 87.8504%
# 召回率: 85.2347%
# Step: 2600 | train loss: 0.0105 | f1分数: 90.1949 | time: 306.6079
# 精度: 90.5330%
# 召回率: 90.9286%
# Step: 2650 | train loss: 0.0464 | f1分数: 89.9599 | time: 312.5354
# 精度: 90.1423%
# 召回率: 90.5918%
# Step: 2700 | train loss: 0.0758 | f1分数: 89.1193 | time: 318.4527
# 精度: 89.6639%
# 召回率: 90.2653%
# Step: 2750 | train loss: 0.0280 | f1分数: 88.5640 | time: 324.3483
# 精度: 88.6991%
# 召回率: 89.7959%
# Step: 2800 | train loss: 0.1699 | f1分数: 89.8363 | time: 330.2276
# 精度: 89.7475%
# 召回率: 90.7143%
# Step: 2850 | train loss: 0.0903 | f1分数: 88.0238 | time: 336.1553
# 精度: 88.1534%
# 召回率: 88.7245%
# Step: 2900 | train loss: 0.0593 | f1分数: 84.2278 | time: 342.0954
# 精度: 89.0510%
# 召回率: 81.8265%
# Step: 2950 | train loss: 0.1652 | f1分数: 82.7019 | time: 348.0455
# 精度: 88.8919%
# 召回率: 79.5510%
# Step: 3000 | train loss: 0.0256 | f1分数: 89.9359 | time: 353.9744
# 精度: 90.3084%
# 召回率: 90.0408%
# Step: 3050 | train loss: 0.1164 | f1分数: 89.9600 | time: 359.9105
# 精度: 90.3483%
# 召回率: 89.9592%
# Step: 3100 | train loss: 0.0565 | f1分数: 90.3619 | time: 365.8221
# 精度: 90.8070%
# 召回率: 90.4184%
# Step: 3150 | train loss: 0.0896 | f1分数: 89.6527 | time: 371.7664
# 精度: 90.3668%
# 召回率: 90.6531%
# Step: 3200 | train loss: 0.0398 | f1分数: 90.2245 | time: 377.6884
# 精度: 90.3769%
# 召回率: 90.5102%
# Step: 3250 | train loss: 0.0228 | f1分数: 90.4815 | time: 383.6257
# 精度: 90.9727%
# 召回率: 90.4082%
# Step: 3300 | train loss: 0.0079 | f1分数: 90.5267 | time: 389.5674
# 精度: 91.2116%
# 召回率: 90.4388%
# Step: 3350 | train loss: 0.0747 | f1分数: 89.7097 | time: 395.5069
# 精度: 90.6859%
# 召回率: 89.3469%
# Step: 3400 | train loss: 0.0213 | f1分数: 91.1476 | time: 401.4420
# 精度: 91.3357%
# 召回率: 91.3061%
# Step: 3450 | train loss: 0.0137 | f1分数: 89.4745 | time: 407.3866
# 精度: 89.8456%
# 召回率: 89.7041%
# Step: 3500 | train loss: 0.0468 | f1分数: 87.9910 | time: 413.3398
# 精度: 89.4640%
# 召回率: 87.8367%
# Step: 3550 | train loss: 0.0730 | f1分数: 88.4798 | time: 419.2630
# 精度: 89.8829%
# 召回率: 88.1633%
# Step: 3600 | train loss: 0.0212 | f1分数: 89.8764 | time: 425.2038
# 精度: 90.5966%
# 召回率: 89.8571%
# Step: 3650 | train loss: 0.0233 | f1分数: 90.3044 | time: 431.1608
# 精度: 90.6075%
# 召回率: 90.7143%
# Step: 3700 | train loss: 0.0096 | f1分数: 90.2796 | time: 437.0885
# 精度: 90.3468%
# 召回率: 90.7653%
# Step: 3750 | train loss: 0.0236 | f1分数: 90.4736 | time: 443.0189
# 精度: 90.4734%
# 召回率: 90.9286%
# Step: 3800 | train loss: 0.0655 | f1分数: 89.6123 | time: 448.9511
# 精度: 89.7415%
# 召回率: 90.4388%
# Step: 3850 | train loss: 0.0467 | f1分数: 89.4783 | time: 454.9041
# 精度: 89.7652%
# 召回率: 90.0510%
# Step: 3900 | train loss: 0.0246 | f1分数: 89.2390 | time: 460.8358
# 精度: 89.7139%
# 召回率: 89.8061%
# Step: 3950 | train loss: 0.0285 | f1分数: 88.6486 | time: 466.7762
# 精度: 89.2091%
# 召回率: 89.2755%
# Step: 4000 | train loss: 0.0763 | f1分数: 88.5584 | time: 472.7381
# 精度: 89.1770%
# 召回率: 88.8980%
# Step: 4050 | train loss: 0.0093 | f1分数: 89.5815 | time: 478.6502
# 精度: 89.7641%
# 召回率: 90.0918%
# Step: 4100 | train loss: 0.0053 | f1分数: 90.7980 | time: 484.6011
# 精度: 91.1260%
# 召回率: 91.4082%
# Step: 4150 | train loss: 0.0102 | f1分数: 85.3682 | time: 490.5186
# 精度: 87.4037%
# 召回率: 84.2041%
# Step: 4200 | train loss: 0.0170 | f1分数: 88.5576 | time: 496.4727
# 精度: 89.3723%
# 召回率: 88.7959%
# Step: 4250 | train loss: 0.0228 | f1分数: 90.0848 | time: 502.3335
# 精度: 90.7739%
# 召回率: 90.0408%
# Step: 4300 | train loss: 0.0758 | f1分数: 89.5063 | time: 508.1078
# 精度: 90.1569%
# 召回率: 89.8571%
# Step: 4350 | train loss: 0.0264 | f1分数: 89.5957 | time: 513.9054
# 精度: 89.8892%
# 召回率: 89.9694%
# Step: 4400 | train loss: 0.0185 | f1分数: 90.5065 | time: 519.7422
# 精度: 91.0679%
# 召回率: 90.4082%
# Step: 4450 | train loss: 0.0035 | f1分数: 90.1130 | time: 525.5723
# 精度: 90.9817%
# 召回率: 89.7245%
# Step: 4500 | train loss: 0.0332 | f1分数: 89.7317 | time: 531.4202
# 精度: 90.7335%
# 召回率: 89.9898%
# Step: 4550 | train loss: 0.0059 | f1分数: 89.6205 | time: 537.2842
# 精度: 90.1345%
# 召回率: 89.5510%
# Step: 4600 | train loss: 0.0133 | f1分数: 89.7850 | time: 543.1549
# 精度: 90.8804%
# 召回率: 89.4082%
# Step: 4650 | train loss: 0.0538 | f1分数: 90.2360 | time: 548.9913
# 精度: 90.9080%
# 召回率: 90.2041%
# Step: 4700 | train loss: 0.0108 | f1分数: 84.7279 | time: 554.8157
# 精度: 89.3959%
# 召回率: 82.3367%
# Step: 4750 | train loss: 0.0123 | f1分数: 90.1478 | time: 560.6596
# 精度: 90.4530%
# 召回率: 90.5510%
# Step: 4800 | train loss: 0.0583 | f1分数: 89.1030 | time: 566.5117
# 精度: 89.9392%
# 召回率: 90.0306%
# Step: 4850 | train loss: 0.0038 | f1分数: 89.4285 | time: 572.3670
# 精度: 89.8575%
# 召回率: 90.2857%
# Step: 4900 | train loss: 0.0101 | f1分数: 87.0286 | time: 578.2201
# 精度: 88.7886%
# 召回率: 86.8469%
# Step: 4950 | train loss: 0.0679 | f1分数: 90.6392 | time: 584.0583
# 精度: 91.0740%
# 召回率: 90.8673%
# Step: 5000 | train loss: 0.0043 | f1分数: 90.1477 | time: 589.9670
# 精度: 90.3966%
# 召回率: 90.3469%
# Step: 5050 | train loss: 0.0019 | f1分数: 90.9423 | time: 595.8011
# 精度: 91.0440%
# 召回率: 91.1939%
# Step: 5100 | train loss: 0.0150 | f1分数: 90.7353 | time: 601.6639
# 精度: 90.7825%
# 召回率: 91.0816%
# Step: 5150 | train loss: 0.0323 | f1分数: 89.7333 | time: 607.5292
# 精度: 90.5470%
# 召回率: 89.6735%
# Step: 5200 | train loss: 0.0093 | f1分数: 88.0675 | time: 613.4001
# 精度: 88.6952%
# 召回率: 88.3776%
# Step: 5250 | train loss: 0.0080 | f1分数: 89.6146 | time: 619.2519
# 精度: 90.3374%
# 召回率: 90.0816%
# Step: 5300 | train loss: 0.0331 | f1分数: 90.6603 | time: 625.0776
# 精度: 90.7317%
# 召回率: 91.1837%
# Step: 5350 | train loss: 0.0202 | f1分数: 90.1366 | time: 630.9282
# 精度: 90.5133%
# 召回率: 90.4592%
# Step: 5400 | train loss: 0.0128 | f1分数: 90.2550 | time: 636.7884
# 精度: 90.4757%
# 召回率: 90.5612%
# Step: 5450 | train loss: 0.0204 | f1分数: 89.7550 | time: 642.6470
# 精度: 90.1309%
# 召回率: 90.0918%
# Step: 5500 | train loss: 0.0256 | f1分数: 90.5616 | time: 648.4950
# 精度: 90.6530%
# 召回率: 90.9184%
# Step: 5550 | train loss: 0.0370 | f1分数: 90.4294 | time: 654.3374
# 精度: 90.8750%
# 召回率: 90.2755%
# Step: 5600 | train loss: 0.0472 | f1分数: 91.0232 | time: 660.2275
# 精度: 91.2669%
# 召回率: 91.1122%
# Step: 5650 | train loss: 0.0096 | f1分数: 90.5336 | time: 666.0870
# 精度: 90.9539%
# 召回率: 90.5000%
# Step: 5700 | train loss: 0.0123 | f1分数: 91.2270 | time: 671.9327
# 精度: 91.4549%
# 召回率: 91.4490%
# Step: 5750 | train loss: 0.0238 | f1分数: 90.3485 | time: 677.8076
# 精度: 90.4886%
# 召回率: 90.6939%
# Step: 5800 | train loss: 0.0231 | f1分数: 90.7776 | time: 683.6689
# 精度: 90.9842%
# 召回率: 90.9286%
# Step: 5850 | train loss: 0.0123 | f1分数: 90.6921 | time: 689.5200
# 精度: 91.0876%
# 召回率: 90.6327%
# Step: 5900 | train loss: 0.0079 | f1分数: 88.7953 | time: 695.3670
# 精度: 89.5889%
# 召回率: 88.8980%
# Step: 5950 | train loss: 0.0064 | f1分数: 89.4591 | time: 701.2163
# 精度: 89.8649%
# 召回率: 90.4082%
# Step: 6000 | train loss: 0.0265 | f1分数: 90.0133 | time: 707.0724
# 精度: 90.0811%
# 召回率: 90.5204%
# Step: 6050 | train loss: 0.0315 | f1分数: 88.9582 | time: 712.9525
# 精度: 89.1200%
# 召回率: 89.9796%
# Step: 6100 | train loss: 0.0143 | f1分数: 90.0841 | time: 718.8158
# 精度: 90.3894%
# 召回率: 91.0816%
# Step: 6150 | train loss: 0.0136 | f1分数: 89.4163 | time: 724.6713
# 精度: 89.7262%
# 召回率: 89.9592%
# Step: 6200 | train loss: 0.0153 | f1分数: 87.5669 | time: 730.5252
# 精度: 89.1275%
# 召回率: 88.3673%
# Step: 6250 | train loss: 0.0398 | f1分数: 88.1578 | time: 736.3585
# 精度: 88.9977%
# 召回率: 88.5204%
# Step: 6300 | train loss: 0.0334 | f1分数: 90.0759 | time: 742.2085
# 精度: 90.4718%
# 召回率: 90.9490%
# Step: 6350 | train loss: 0.0040 | f1分数: 89.4658 | time: 748.0810
# 精度: 89.8126%
# 召回率: 90.2143%
# Step: 6400 | train loss: 0.0075 | f1分数: 89.3078 | time: 753.9352
# 精度: 89.5792%
# 召回率: 89.9286%
# Step: 6450 | train loss: 0.0243 | f1分数: 90.4978 | time: 759.7638
# 精度: 90.7294%
# 召回率: 90.6429%
# Step: 6500 | train loss: 0.0124 | f1分数: 90.1568 | time: 765.5914
# 精度: 90.6259%
# 召回率: 90.1122%
# Step: 6550 | train loss: 0.0099 | f1分数: 90.4481 | time: 771.4471
# 精度: 90.7089%
# 召回率: 90.7245%
# Step: 6600 | train loss: 0.0148 | f1分数: 91.1341 | time: 777.3016
# 精度: 91.1096%
# 召回率: 91.5204%
# Step: 6650 | train loss: 0.0084 | f1分数: 89.7988 | time: 783.1359
# 精度: 90.2348%
# 召回率: 90.0306%
# Step: 6700 | train loss: 0.0060 | f1分数: 90.4732 | time: 788.9651
# 精度: 90.6871%
# 召回率: 90.6224%
# Step: 6750 | train loss: 0.0003 | f1分数: 90.9920 | time: 794.8035
# 精度: 91.0616%
# 召回率: 91.1939%
# Step: 6800 | train loss: 0.0263 | f1分数: 90.1958 | time: 800.6376
# 精度: 90.1741%
# 召回率: 90.9082%
# Step: 6850 | train loss: 0.0002 | f1分数: 90.5197 | time: 806.4938
# 精度: 90.7064%
# 召回率: 90.6429%
# Step: 6900 | train loss: 0.0024 | f1分数: 89.9627 | time: 812.3554
# 精度: 90.3321%
# 召回率: 90.1735%
# Step: 6950 | train loss: 0.0092 | f1分数: 90.7251 | time: 818.1945
# 精度: 90.7609%
# 召回率: 91.2041%
# Step: 7000 | train loss: 0.0023 | f1分数: 89.9181 | time: 824.0573
# 精度: 89.9649%
# 召回率: 90.6939%
# Step: 7050 | train loss: 0.0040 | f1分数: 90.8096 | time: 829.9003
# 精度: 91.1851%
# 召回率: 90.7959%
# Step: 7100 | train loss: 0.0006 | f1分数: 90.5319 | time: 835.7675
# 精度: 90.9037%
# 召回率: 90.6939%
# Step: 7150 | train loss: 0.0129 | f1分数: 90.5849 | time: 841.6205
# 精度: 91.0060%
# 召回率: 90.5816%
# Step: 7200 | train loss: 0.0005 | f1分数: 90.4320 | time: 847.4776
# 精度: 90.7656%
# 召回率: 90.4592%
# Step: 7250 | train loss: 0.0024 | f1分数: 90.3359 | time: 853.3266
# 精度: 90.5196%
# 召回率: 90.8469%
# Step: 7300 | train loss: 0.0042 | f1分数: 90.2934 | time: 859.1889
# 精度: 90.8100%
# 召回率: 90.2653%
# Step: 7350 | train loss: 0.0103 | f1分数: 90.6569 | time: 865.0221
# 精度: 90.6668%
# 召回率: 91.2041%
# Step: 7400 | train loss: 0.0044 | f1分数: 90.4257 | time: 870.8691
# 精度: 90.6895%
# 召回率: 90.8163%
# Step: 7450 | train loss: 0.0158 | f1分数: 90.0770 | time: 876.7035
# 精度: 90.2375%
# 召回率: 90.8163%
# Step: 7500 | train loss: 0.0013 | f1分数: 89.7236 | time: 882.5174
# 精度: 89.9935%
# 召回率: 90.6327%
# Step: 7550 | train loss: 0.0068 | f1分数: 88.9702 | time: 888.3577
# 精度: 89.6321%
# 召回率: 89.9898%
# Step: 7600 | train loss: 0.0201 | f1分数: 90.2731 | time: 894.1868
# 精度: 90.5666%
# 召回率: 90.6122%
# Step: 7650 | train loss: 0.0040 | f1分数: 90.1798 | time: 900.0384
# 精度: 90.7739%
# 召回率: 90.3673%
# Step: 7700 | train loss: 0.0054 | f1分数: 90.5720 | time: 905.9412
# 精度: 90.7607%
# 召回率: 91.0204%
# Step: 7750 | train loss: 0.0066 | f1分数: 90.5410 | time: 911.8545
# 精度: 90.7580%
# 召回率: 90.6429%
# Step: 7800 | train loss: 0.0199 | f1分数: 90.3182 | time: 917.6989
# 精度: 90.4338%
# 召回率: 90.6837%
# Step: 7850 | train loss: 0.0139 | f1分数: 90.8077 | time: 923.5669
# 精度: 90.8964%
# 召回率: 91.2245%
# Step: 7900 | train loss: 0.0008 | f1分数: 90.1174 | time: 929.3881
# 精度: 90.3146%
# 召回率: 90.8265%
# Step: 7950 | train loss: 0.0184 | f1分数: 90.4171 | time: 935.2175
# 精度: 90.8134%
# 召回率: 90.8571%
# Step: 8000 | train loss: 0.0393 | f1分数: 82.3784 | time: 941.0671
# 精度: 88.3088%
# 召回率: 79.0000%
# Step: 8050 | train loss: 0.0163 | f1分数: 89.8472 | time: 946.9162
# 精度: 90.3308%
# 召回率: 90.1327%
# Step: 8100 | train loss: 0.0260 | f1分数: 88.5515 | time: 952.7608
# 精度: 89.2063%
# 召回率: 89.8265%
# Step: 8150 | train loss: 0.0281 | f1分数: 88.4884 | time: 958.6092
# 精度: 88.9531%
# 召回率: 89.4286%
# Step: 8200 | train loss: 0.0416 | f1分数: 88.7110 | time: 964.4511
# 精度: 90.1365%
# 召回率: 88.9796%
# Step: 8250 | train loss: 0.0124 | f1分数: 89.8843 | time: 970.3069
# 精度: 90.5051%
# 召回率: 90.0102%
# Step: 8300 | train loss: 0.0155 | f1分数: 89.6606 | time: 976.1787
# 精度: 90.3904%
# 召回率: 90.0510%
# Step: 8350 | train loss: 0.0286 | f1分数: 89.2648 | time: 982.0262
# 精度: 90.1665%
# 召回率: 89.2449%
# Step: 8400 | train loss: 0.0035 | f1分数: 89.8667 | time: 987.8932
# 精度: 90.6033%
# 召回率: 89.7755%
# Step: 8450 | train loss: 0.0475 | f1分数: 90.6419 | time: 993.7632
# 精度: 90.6567%
# 召回率: 91.0306%
# Step: 8500 | train loss: 0.0135 | f1分数: 89.3933 | time: 999.6219
# 精度: 89.5697%
# 召回率: 89.8367%
# Step: 8550 | train loss: 0.0038 | f1分数: 90.4790 | time: 1005.4742
# 精度: 90.6810%
# 召回率: 91.0510%
# Step: 8600 | train loss: 0.0139 | f1分数: 89.7370 | time: 1011.3614
# 精度: 89.9516%
# 召回率: 90.5000%
# Step: 8650 | train loss: 0.0867 | f1分数: 89.8657 | time: 1017.1999
# 精度: 90.3393%
# 召回率: 90.1939%
# Step: 8700 | train loss: 0.0000 | f1分数: 90.7283 | time: 1023.0605
# 精度: 90.9736%
# 召回率: 91.0306%
# Step: 8750 | train loss: 0.0032 | f1分数: 89.7494 | time: 1028.9161
# 精度: 90.1584%
# 召回率: 89.9898%
# Step: 8800 | train loss: 0.0063 | f1分数: 90.7012 | time: 1034.7532
# 精度: 91.1024%
# 召回率: 90.7755%
# Step: 8850 | train loss: 0.0027 | f1分数: 90.0509 | time: 1040.6176
# 精度: 90.6206%
# 召回率: 90.2959%
# Step: 8900 | train loss: 0.0002 | f1分数: 90.0051 | time: 1046.4772
# 精度: 90.1802%
# 召回率: 90.3980%
# Step: 8950 | train loss: 0.0061 | f1分数: 90.0548 | time: 1052.3470
# 精度: 90.3181%
# 召回率: 90.5408%
# Step: 9000 | train loss: 0.0390 | f1分数: 90.2719 | time: 1058.2056
# 精度: 90.4330%
# 召回率: 90.7857%
# Step: 9050 | train loss: 0.0006 | f1分数: 90.4947 | time: 1064.0401
# 精度: 90.7084%
# 召回率: 90.9184%
# Step: 9100 | train loss: 0.0030 | f1分数: 90.7797 | time: 1069.8724
# 精度: 91.0584%
# 召回率: 90.9592%
# Step: 9150 | train loss: 0.0017 | f1分数: 90.4594 | time: 1075.7225
# 精度: 90.5833%
# 召回率: 90.7653%
# Step: 9200 | train loss: 0.0059 | f1分数: 90.2807 | time: 1081.5809
# 精度: 90.6476%
# 召回率: 90.2755%
# Step: 9250 | train loss: 0.0025 | f1分数: 91.0798 | time: 1087.3944
# 精度: 91.1083%
# 召回率: 91.4388%
# Step: 9300 | train loss: 0.0005 | f1分数: 90.8948 | time: 1093.2380
# 精度: 90.9804%
# 召回率: 91.2245%
# Step: 9350 | train loss: 0.0024 | f1分数: 90.5328 | time: 1099.0973
# 精度: 90.7650%
# 召回率: 91.2041%
# Step: 9400 | train loss: 0.0001 | f1分数: 90.9363 | time: 1104.9202
# 精度: 91.1206%
# 召回率: 91.0816%
# Step: 9450 | train loss: 0.0023 | f1分数: 90.4651 | time: 1110.8272
# 精度: 90.6689%
# 召回率: 90.9184%
# Step: 9500 | train loss: 0.0001 | f1分数: 90.6707 | time: 1116.6707
# 精度: 90.7210%
# 召回率: 91.0408%
# Step: 9550 | train loss: 0.0002 | f1分数: 90.4889 | time: 1122.5137
# 精度: 90.7155%
# 召回率: 90.7449%
# Step: 9600 | train loss: 0.0003 | f1分数: 90.4409 | time: 1128.3749
# 精度: 90.6852%
# 召回率: 91.0102%
# Step: 9650 | train loss: 0.0004 | f1分数: 90.3384 | time: 1134.2117
# 精度: 90.4126%
# 召回率: 90.9082%
# Step: 9700 | train loss: 0.0038 | f1分数: 90.6848 | time: 1140.0766
# 精度: 90.8304%
# 召回率: 90.9592%
# Step: 9750 | train loss: 0.0001 | f1分数: 90.4266 | time: 1145.9406
# 精度: 90.5955%
# 召回率: 90.7857%
# Step: 9800 | train loss: 0.0010 | f1分数: 90.5122 | time: 1151.7946
# 精度: 90.5999%
# 召回率: 90.9694%
# Step: 9850 | train loss: 0.0003 | f1分数: 91.0399 | time: 1157.6595
# 精度: 91.1833%
# 召回率: 91.3163%
# Step: 9900 | train loss: 0.0015 | f1分数: 83.3955 | time: 1163.5132
# 精度: 88.7445%
# 召回率: 80.2857%
# Step: 9950 | train loss: 0.0112 | f1分数: 90.4850 | time: 1169.3753
# 精度: 90.8567%
# 召回率: 90.6429%
# Step: 10000 | train loss: 0.0130 | f1分数: 90.9957 | time: 1175.2322
# 精度: 91.1069%
# 召回率: 91.2041%
# Step: 10050 | train loss: 0.0258 | f1分数: 89.6120 | time: 1181.0860
# 精度: 89.7364%
# 召回率: 90.3163%
# Step: 10100 | train loss: 0.2367 | f1分数: 86.1711 | time: 1186.9527
# 精度: 87.5901%
# 召回率: 86.5918%
# Step: 10150 | train loss: 0.0046 | f1分数: 86.0228 | time: 1192.7792
# 精度: 88.0146%
# 召回率: 85.2857%
# Step: 10200 | train loss: 0.0057 | f1分数: 90.1594 | time: 1198.6112
# 精度: 90.3828%
# 召回率: 90.6837%
# Step: 10250 | train loss: 0.0486 | f1分数: 88.5773 | time: 1204.4643
# 精度: 89.2215%
# 召回率: 89.4796%
# Step: 10300 | train loss: 0.0071 | f1分数: 89.7857 | time: 1210.3530
# 精度: 90.0449%
# 召回率: 90.0306%
# Step: 10350 | train loss: 0.0334 | f1分数: 90.1734 | time: 1216.1862
# 精度: 90.7118%
# 召回率: 90.5612%
# Step: 10400 | train loss: 0.0122 | f1分数: 90.8292 | time: 1222.0330
# 精度: 91.0010%
# 召回率: 91.0612%
# Step: 10450 | train loss: 0.0083 | f1分数: 90.5871 | time: 1227.8962
# 精度: 90.6858%
# 召回率: 90.9286%
# Step: 10500 | train loss: 0.0010 | f1分数: 91.0144 | time: 1233.7456
# 精度: 91.0510%
# 召回率: 91.4490%
# Step: 10550 | train loss: 0.0001 | f1分数: 91.1052 | time: 1239.5768
# 精度: 91.1224%
# 召回率: 91.5918%
# Step: 10600 | train loss: 0.0001 | f1分数: 91.1898 | time: 1245.4172
# 精度: 91.1991%
# 召回率: 91.5510%
# Step: 10650 | train loss: 0.0002 | f1分数: 91.2143 | time: 1251.2893
# 精度: 91.2362%
# 召回率: 91.6429%
# Step: 10700 | train loss: 0.0019 | f1分数: 91.2324 | time: 1257.1404
# 精度: 91.2816%
# 召回率: 91.6122%
# Step: 10750 | train loss: 0.0000 | f1分数: 91.0980 | time: 1262.9766
# 精度: 91.1051%
# 召回率: 91.4898%
# Step: 10800 | train loss: 0.0128 | f1分数: 91.1482 | time: 1268.8335
# 精度: 91.2002%
# 召回率: 91.5204%
# Step: 10850 | train loss: 0.0023 | f1分数: 91.0694 | time: 1274.7054
# 精度: 91.1205%
# 召回率: 91.6224%
# Step: 10900 | train loss: 0.0001 | f1分数: 90.9901 | time: 1280.5563
# 精度: 91.0059%
# 召回率: 91.3163%
# Step: 10950 | train loss: 0.0003 | f1分数: 90.9450 | time: 1286.4072
# 精度: 90.9658%
# 召回率: 91.4184%
# Step: 11000 | train loss: 0.0082 | f1分数: 90.9683 | time: 1292.2789
# 精度: 91.0407%
# 召回率: 91.3265%
# Step: 11050 | train loss: 0.0000 | f1分数: 91.0297 | time: 1298.1205
# 精度: 91.0544%
# 召回率: 91.4694%
# Step: 11100 | train loss: 0.0016 | f1分数: 89.4053 | time: 1303.9631
# 精度: 89.7959%
# 召回率: 90.2449%
# Step: 11150 | train loss: 0.0006 | f1分数: 90.8479 | time: 1309.8571
# 精度: 90.9034%
# 召回率: 91.3163%
# Step: 11200 | train loss: 0.0052 | f1分数: 89.9749 | time: 1315.7212
# 精度: 90.8544%
# 召回率: 89.8571%
# Step: 11250 | train loss: 0.0222 | f1分数: 90.1197 | time: 1321.6020
# 精度: 91.3008%
# 召回率: 89.8367%
# Step: 11300 | train loss: 0.0065 | f1分数: 90.3463 | time: 1327.4446
# 精度: 90.8430%
# 召回率: 90.5204%
# Step: 11350 | train loss: 0.0046 | f1分数: 89.7459 | time: 1333.2802
# 精度: 90.0682%
# 召回率: 90.3878%
# Step: 11400 | train loss: 0.0024 | f1分数: 89.6767 | time: 1339.1211
# 精度: 90.4867%
# 召回率: 89.8980%
# Step: 11450 | train loss: 0.0061 | f1分数: 89.2025 | time: 1344.9695
# 精度: 89.4652%
# 召回率: 89.7857%
# Step: 11500 | train loss: 0.0067 | f1分数: 90.2585 | time: 1350.8217
# 精度: 90.5685%
# 召回率: 90.4184%
# Step: 11550 | train loss: 0.0007 | f1分数: 90.5176 | time: 1356.6539
# 精度: 90.7746%
# 召回率: 90.9388%
# Step: 11600 | train loss: 0.0040 | f1分数: 89.5604 | time: 1362.5039
# 精度: 89.6862%
# 召回率: 90.2857%
# Step: 11650 | train loss: 0.0008 | f1分数: 90.2377 | time: 1368.3525
# 精度: 90.6253%
# 召回率: 90.6224%
# Step: 11700 | train loss: 0.0019 | f1分数: 89.7346 | time: 1374.2140
# 精度: 89.9889%
# 召回率: 90.2857%
# Step: 11750 | train loss: 0.0065 | f1分数: 90.4672 | time: 1380.0800
# 精度: 90.6316%
# 召回率: 90.8571%
# Step: 11800 | train loss: 0.0075 | f1分数: 89.6528 | time: 1385.9516
# 精度: 89.9777%
# 召回率: 90.3469%
# Step: 11850 | train loss: 0.0014 | f1分数: 90.3170 | time: 1391.8159
# 精度: 90.3861%
# 召回率: 90.9592%
# Step: 11900 | train loss: 0.0018 | f1分数: 90.5909 | time: 1397.6564
# 精度: 90.6495%
# 召回率: 91.0408%
# Step: 11950 | train loss: 0.0006 | f1分数: 91.1548 | time: 1403.5104
# 精度: 91.2761%
# 召回率: 91.3673%
# Step: 12000 | train loss: 0.0044 | f1分数: 90.4668 | time: 1409.4196
# 精度: 90.4849%
# 召回率: 90.9898%
# Step: 12050 | train loss: 0.0007 | f1分数: 88.3040 | time: 1415.2267
# 精度: 89.4067%
# 召回率: 88.1531%
# Step: 12100 | train loss: 0.0217 | f1分数: 89.9705 | time: 1421.0918
# 精度: 90.1829%
# 召回率: 90.1633%
# Step: 12150 | train loss: 0.0136 | f1分数: 89.0185 | time: 1426.9708
# 精度: 89.0812%
# 召回率: 89.5306%
# Step: 12200 | train loss: 0.0066 | f1分数: 89.8113 | time: 1432.8395
# 精度: 90.0797%
# 召回率: 90.5714%
# Step: 12250 | train loss: 0.0105 | f1分数: 89.7305 | time: 1438.6659
# 精度: 90.0099%
# 召回率: 90.2347%
# Step: 12300 | train loss: 0.0060 | f1分数: 89.7807 | time: 1444.5154
# 精度: 89.9738%
# 召回率: 90.3980%
# Step: 12350 | train loss: 0.0009 | f1分数: 90.0891 | time: 1450.3784
# 精度: 90.1379%
# 召回率: 90.6020%
# Step: 12400 | train loss: 0.0004 | f1分数: 90.8261 | time: 1456.2098
# 精度: 91.0594%
# 召回率: 90.9796%
# Step: 12450 | train loss: 0.0011 | f1分数: 91.0828 | time: 1462.0693
# 精度: 91.1540%
# 召回率: 91.4184%
# Step: 12500 | train loss: 0.0001 | f1分数: 91.0183 | time: 1467.9229
# 精度: 91.0954%
# 召回率: 91.3673%
# Step: 12550 | train loss: 0.0000 | f1分数: 91.1580 | time: 1473.7732
# 精度: 91.2377%
# 召回率: 91.4286%
# Step: 12600 | train loss: 0.0004 | f1分数: 91.0817 | time: 1479.6241
# 精度: 91.1719%
# 召回率: 91.3673%
# Step: 12650 | train loss: 0.0002 | f1分数: 91.0468 | time: 1485.4950
# 精度: 91.0985%
# 召回率: 91.3571%
# Step: 12700 | train loss: 0.0001 | f1分数: 91.0846 | time: 1491.3363
# 精度: 91.2246%
# 召回率: 91.2959%
# Step: 12750 | train loss: 0.0001 | f1分数: 90.9004 | time: 1497.2124
# 精度: 90.9494%
# 召回率: 91.3980%
# Step: 12800 | train loss: 0.0000 | f1分数: 91.0824 | time: 1503.0519
# 精度: 91.1798%
# 召回率: 91.3367%
# Step: 12850 | train loss: 0.0004 | f1分数: 91.0269 | time: 1508.9281
# 精度: 91.1226%
# 召回率: 91.2755%
# Step: 12900 | train loss: 0.0001 | f1分数: 91.1101 | time: 1514.7601
# 精度: 91.1794%
# 召回率: 91.5204%
# Step: 12950 | train loss: 0.0001 | f1分数: 90.7961 | time: 1520.6148
# 精度: 91.0739%
# 召回率: 90.8571%
# Step: 13000 | train loss: 0.0071 | f1分数: 90.7569 | time: 1526.4633
# 精度: 91.0959%
# 召回率: 91.0000%
# Step: 13050 | train loss: 0.0086 | f1分数: 90.0096 | time: 1532.3306
# 精度: 90.1145%
# 召回率: 90.4286%
# Step: 13100 | train loss: 0.0127 | f1分数: 89.1062 | time: 1538.1858
# 精度: 89.9903%
# 召回率: 89.7245%
# Step: 13150 | train loss: 0.0189 | f1分数: 88.9587 | time: 1544.0187
# 精度: 90.0707%
# 召回率: 88.6939%
# Step: 13200 | train loss: 0.0146 | f1分数: 90.3681 | time: 1549.8573
# 精度: 90.6924%
# 召回率: 90.5510%
# Step: 13250 | train loss: 0.0421 | f1分数: 90.1143 | time: 1555.7157
# 精度: 90.6224%
# 召回率: 90.3673%
# Step: 13300 | train loss: 0.0221 | f1分数: 90.1949 | time: 1561.5638
# 精度: 90.4205%
# 召回率: 90.1735%
# Step: 13350 | train loss: 0.0430 | f1分数: 89.2998 | time: 1567.4084
# 精度: 89.8127%
# 召回率: 89.6224%
# Step: 13400 | train loss: 0.0029 | f1分数: 90.5913 | time: 1573.2482
# 精度: 91.0769%
# 召回率: 90.4388%
# Step: 13450 | train loss: 0.0121 | f1分数: 89.6050 | time: 1579.0909
# 精度: 90.2139%
# 召回率: 89.9388%
# Step: 13500 | train loss: 0.0133 | f1分数: 90.5686 | time: 1584.9348
# 精度: 90.7267%
# 召回率: 91.1531%
# Step: 13550 | train loss: 0.0005 | f1分数: 90.5364 | time: 1590.8037
# 精度: 90.7231%
# 召回率: 90.6837%
# Step: 13600 | train loss: 0.0006 | f1分数: 90.8822 | time: 1596.6902
# 精度: 91.0622%
# 召回率: 91.0102%
# Step: 13650 | train loss: 0.0003 | f1分数: 90.8608 | time: 1602.5401
# 精度: 91.0393%
# 召回率: 91.2857%
# Step: 13700 | train loss: 0.0002 | f1分数: 90.9179 | time: 1608.4129
# 精度: 91.0480%
# 召回率: 91.1020%
# Step: 13750 | train loss: 0.0038 | f1分数: 91.0437 | time: 1614.2773
# 精度: 91.1860%
# 召回率: 91.1633%
# Step: 13800 | train loss: 0.0069 | f1分数: 90.6521 | time: 1620.1173
# 精度: 90.7334%
# 召回率: 90.9592%
# Step: 13850 | train loss: 0.0005 | f1分数: 90.8801 | time: 1625.9565
# 精度: 90.8928%
# 召回率: 91.1224%
# Step: 13900 | train loss: 0.0008 | f1分数: 90.4083 | time: 1631.7955
# 精度: 90.4423%
# 召回率: 90.8061%