import tensorflow as tf
import numpy as np
from sklearn import metrics
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
num_units_gru = 128
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


def ConvGRU(xs, is_training):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
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
    # pool3 = tf.layers.max_pooling2d(conv3, [2, 4], [2, 4], padding='same')
    shape = conv3.get_shape().as_list()
    print('gru input shape: {}'.format(shape))
    flat = tf.reshape(conv3, [-1, shape[1], shape[2] * shape[3]])
    # conv1 = tf.layers.conv2d(xs, 16, [5, 1], 1, 'valid')
    # conv2 = tf.layers.conv2d(conv1, 32, [5, 1], 1, 'valid')
    # conv3 = tf.layers.conv2d(conv2, 64, [5, 1], 1, 'valid')
    # pool = tf.layers.max_pooling2d(conv3, [2, 1], [2, 1], padding='same')
    # # conv4 = tf.layers.conv2d(conv3, 64, 5, 1, 'valid')
    # shape = pool.get_shape().as_list()
    # print('gru input shape: {}'.format(shape))
    # flat = tf.reshape(pool, [-1, shape[1], shape[2] * shape[3]])
    rnn_cell_1 = tf.contrib.rnn.GRUCell(num_units_gru)
    rnn_cell_2 = tf.contrib.rnn.GRUCell(num_units_gru)
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

output = ConvGRU(xs, is_training)

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
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())  # the local var is for accuracy_op
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
            # pred = sess.run(argmax_pred, feed_dict={xs: X_test, is_training: False})
            # accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_
                  # , '| test accuracy: %.4f' % accuracy_
                  # , '| f1分数: %.4f' % (100 * metrics.f1_score(argmax_y, pred, average='weighted'))
                  , '| f1分数: %.4f' % (100 * metrics.f1_score(test_true, test_pred, average='weighted'))
                  , '| time: %.4f' % (time.time() - start_time))
            print("精度: {:.4f}%".format(100 * metrics.precision_score(test_true, test_pred, average="weighted")))
            print("召回率: {:.4f}%".format(100 * metrics.recall_score(test_true, test_pred, average="weighted")))
        # if step % 2000 == 0 and op == 1:
        #     np.save("./figure/OPP1/gru_losses_5.npy", train_losses)
        #     np.save("./figure/OPP1/gru_time_5.npy", Time)
        #     print('损失已保存为numpy文件')
        # if step % 100 == 0 and op == 1 and accuracy_>min_acc:
        #     saver.save(sess, "./OPPmodel/ConvGRU_model")
        #     min_acc = accuracy_
        #     print('ConvGRU模型保存成功')
        # if step % 1000 == 0:
        #     indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
        #     plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        #     plt.show()
        step += 1

# padding = same
# Step: 50 | train loss: 2.7947 | f1分数: 75.6440 | time: 4.1222
# Step: 100 | train loss: 0.8259 | f1分数: 75.6440 | time: 6.8689
# Step: 150 | train loss: 1.0250 | f1分数: 75.6440 | time: 9.5725
# Step: 200 | train loss: 0.7966 | f1分数: 75.6440 | time: 12.2369
# Step: 250 | train loss: 1.0144 | f1分数: 75.6440 | time: 14.9094
# Step: 300 | train loss: 2.2637 | f1分数: 75.7554 | time: 17.6007
# Step: 350 | train loss: 0.9334 | f1分数: 66.2953 | time: 20.3137
# Step: 400 | train loss: 1.3037 | f1分数: 77.1778 | time: 22.9684
# Step: 450 | train loss: 0.7035 | f1分数: 77.5552 | time: 25.6482
# Step: 500 | train loss: 1.2342 | f1分数: 79.2130 | time: 28.3147
# Step: 550 | train loss: 1.1315 | f1分数: 80.1537 | time: 31.0118
# Step: 600 | train loss: 0.4397 | f1分数: 81.2352 | time: 33.6928
# Step: 650 | train loss: 0.4889 | f1分数: 82.3525 | time: 36.3814
# Step: 700 | train loss: 0.8464 | f1分数: 81.7490 | time: 39.0642
# Step: 750 | train loss: 0.4047 | f1分数: 81.8808 | time: 41.7537
# Step: 800 | train loss: 0.6239 | f1分数: 80.5842 | time: 44.4244
# Step: 850 | train loss: 0.5540 | f1分数: 80.7990 | time: 47.1001
# Step: 900 | train loss: 0.7032 | f1分数: 81.0280 | time: 49.7558
# Step: 950 | train loss: 0.5214 | f1分数: 82.6175 | time: 52.4702
# Step: 1000 | train loss: 0.6315 | f1分数: 85.1263 | time: 55.1771
# Step: 1050 | train loss: 0.4342 | f1分数: 85.4805 | time: 57.8632
# Step: 1100 | train loss: 0.5083 | f1分数: 85.1349 | time: 60.5339
# Step: 1150 | train loss: 0.5696 | f1分数: 85.2704 | time: 63.1924
# Step: 1200 | train loss: 0.4939 | f1分数: 85.9883 | time: 65.8663
# Step: 1250 | train loss: 0.5507 | f1分数: 85.5915 | time: 68.5379
# Step: 1300 | train loss: 0.6180 | f1分数: 84.6034 | time: 71.1776
# Step: 1350 | train loss: 0.4672 | f1分数: 84.6300 | time: 73.8756
# Step: 1400 | train loss: 0.1164 | f1分数: 85.2833 | time: 76.5649
# Step: 1450 | train loss: 0.2304 | f1分数: 84.9382 | time: 79.2557
# Step: 1500 | train loss: 0.2662 | f1分数: 85.7532 | time: 81.9488
# Step: 1550 | train loss: 0.2430 | f1分数: 86.2857 | time: 84.6196
# Step: 1600 | train loss: 0.2297 | f1分数: 86.9626 | time: 87.3048
# Step: 1650 | train loss: 0.3372 | f1分数: 86.8177 | time: 89.9579
# Step: 1700 | train loss: 0.2343 | f1分数: 86.6548 | time: 92.6465
# Step: 1750 | train loss: 0.2978 | f1分数: 87.5343 | time: 95.3351
# Step: 1800 | train loss: 0.3090 | f1分数: 88.4482 | time: 98.0100
# Step: 1850 | train loss: 0.2958 | f1分数: 87.1994 | time: 100.6993
# Step: 1900 | train loss: 0.2480 | f1分数: 87.2110 | time: 103.3884
# Step: 1950 | train loss: 0.2372 | f1分数: 86.7000 | time: 106.0806
# Step: 2000 | train loss: 0.2845 | f1分数: 87.7702 | time: 108.7737
# Step: 2050 | train loss: 0.3032 | f1分数: 87.4741 | time: 111.4462
# Step: 2100 | train loss: 0.2164 | f1分数: 87.5040 | time: 114.1556
# Step: 2150 | train loss: 0.2171 | f1分数: 87.7580 | time: 116.8126
# Step: 2200 | train loss: 0.2591 | f1分数: 85.4924 | time: 119.5136
# Step: 2250 | train loss: 0.2249 | f1分数: 87.2080 | time: 122.1943
# Step: 2300 | train loss: 0.2810 | f1分数: 87.7227 | time: 124.8754
# Step: 2350 | train loss: 0.1668 | f1分数: 88.6544 | time: 127.4892
# Step: 2400 | train loss: 0.2062 | f1分数: 88.2804 | time: 130.1673
# Step: 2450 | train loss: 0.1293 | f1分数: 87.1696 | time: 132.8750
# Step: 2500 | train loss: 0.2224 | f1分数: 88.4926 | time: 135.5116
# Step: 2550 | train loss: 0.0823 | f1分数: 80.9023 | time: 138.2110
# Step: 2600 | train loss: 0.1045 | f1分数: 84.9642 | time: 140.8899
# Step: 2650 | train loss: 0.2063 | f1分数: 88.5853 | time: 143.5563
# Step: 2700 | train loss: 0.1907 | f1分数: 87.2026 | time: 146.2635
# Step: 2750 | train loss: 0.0764 | f1分数: 86.1322 | time: 148.9599
# Step: 2800 | train loss: 0.1426 | f1分数: 86.1614 | time: 151.6507
# Step: 2850 | train loss: 0.1890 | f1分数: 86.0620 | time: 154.3814
# Step: 2900 | train loss: 0.1330 | f1分数: 85.3335 | time: 157.0510
# Step: 2950 | train loss: 0.2721 | f1分数: 84.5101 | time: 159.7205
# Step: 3000 | train loss: 0.0792 | f1分数: 86.9671 | time: 162.4314
# Step: 3050 | train loss: 0.2126 | f1分数: 88.0900 | time: 165.1167
# Step: 3100 | train loss: 0.1556 | f1分数: 89.3770 | time: 167.7866
# Step: 3150 | train loss: 0.1082 | f1分数: 89.1181 | time: 170.4954
# Step: 3200 | train loss: 0.1194 | f1分数: 81.0543 | time: 173.1736
# Step: 3250 | train loss: 0.1023 | f1分数: 89.0957 | time: 175.8438
# Step: 3300 | train loss: 0.0956 | f1分数: 88.8849 | time: 178.5043
# Step: 3350 | train loss: 0.1740 | f1分数: 86.7981 | time: 181.1604
# Step: 3400 | train loss: 0.1256 | f1分数: 88.2924 | time: 183.8435
# Step: 3450 | train loss: 0.1502 | f1分数: 86.4193 | time: 186.5357
# Step: 3500 | train loss: 0.1023 | f1分数: 87.4077 | time: 189.2377
# Step: 3550 | train loss: 0.1306 | f1分数: 89.1098 | time: 191.8950
# Step: 3600 | train loss: 0.0687 | f1分数: 88.6056 | time: 194.5727
# Step: 3650 | train loss: 0.0962 | f1分数: 88.4709 | time: 197.2689
# Step: 3700 | train loss: 0.0808 | f1分数: 88.9764 | time: 199.9271
# Step: 3750 | train loss: 0.1551 | f1分数: 88.9200 | time: 202.6126
# Step: 3800 | train loss: 0.1682 | f1分数: 89.2213 | time: 205.2935
# Step: 3850 | train loss: 0.0865 | f1分数: 88.3347 | time: 207.9689
# Step: 3900 | train loss: 0.0828 | f1分数: 89.9143 | time: 210.6695
# Step: 3950 | train loss: 0.0640 | f1分数: 88.8647 | time: 213.3890
# Step: 4000 | train loss: 0.1200 | f1分数: 88.9696 | time: 216.0691
# Step: 4050 | train loss: 0.0230 | f1分数: 89.2526 | time: 218.7816
# Step: 4100 | train loss: 0.0511 | f1分数: 87.8979 | time: 221.4743
# Step: 4150 | train loss: 0.0789 | f1分数: 86.5727 | time: 224.1729
# Step: 4200 | train loss: 0.0469 | f1分数: 86.3258 | time: 226.8554
# Step: 4250 | train loss: 0.0407 | f1分数: 86.8461 | time: 229.5489
# Step: 4300 | train loss: 0.0672 | f1分数: 88.4436 | time: 232.2638
# Step: 4350 | train loss: 0.0397 | f1分数: 89.5016 | time: 234.9752
# Step: 4400 | train loss: 0.0398 | f1分数: 89.4369 | time: 237.6418
# Step: 4450 | train loss: 0.0262 | f1分数: 88.2220 | time: 240.2898
# Step: 4500 | train loss: 0.0514 | f1分数: 88.3199 | time: 242.9486
# Step: 4550 | train loss: 0.0254 | f1分数: 88.1609 | time: 245.5966
# Step: 4600 | train loss: 0.0618 | f1分数: 88.7498 | time: 248.2687
# Step: 4650 | train loss: 0.0494 | f1分数: 88.5224 | time: 250.9412
# Step: 4700 | train loss: 0.0431 | f1分数: 88.4834 | time: 253.6563
# Step: 4750 | train loss: 0.0735 | f1分数: 88.4585 | time: 256.3557
# Step: 4800 | train loss: 0.1038 | f1分数: 87.7938 | time: 259.0452
# Step: 4850 | train loss: 0.0214 | f1分数: 87.9247 | time: 261.7311
# Step: 4900 | train loss: 0.0329 | f1分数: 88.3904 | time: 264.4209
# Step: 4950 | train loss: 0.0479 | f1分数: 87.5412 | time: 267.1204
# Step: 5000 | train loss: 0.0360 | f1分数: 88.5168 | time: 269.7657
# Step: 5050 | train loss: 0.0199 | f1分数: 88.6229 | time: 272.4369
# Step: 5100 | train loss: 0.0324 | f1分数: 88.5182 | time: 275.0944
# Step: 5150 | train loss: 0.0703 | f1分数: 89.0361 | time: 277.8209
# Step: 5200 | train loss: 0.0456 | f1分数: 88.8459 | time: 280.5103
# Step: 5250 | train loss: 0.0247 | f1分数: 89.5615 | time: 283.1993
# Step: 5300 | train loss: 0.0585 | f1分数: 88.9954 | time: 285.8565
# Step: 5350 | train loss: 0.0615 | f1分数: 89.0351 | time: 288.5088
# Step: 5400 | train loss: 0.0658 | f1分数: 88.8396 | time: 291.1805
# Step: 5450 | train loss: 0.0295 | f1分数: 89.3390 | time: 293.8204
# Step: 5500 | train loss: 0.0517 | f1分数: 88.8375 | time: 296.4641
# Step: 5550 | train loss: 0.0607 | f1分数: 88.8666 | time: 299.1055
# Step: 5600 | train loss: 0.0964 | f1分数: 88.3050 | time: 301.7116
# Step: 5650 | train loss: 0.0219 | f1分数: 88.5895 | time: 304.3639
# Step: 5700 | train loss: 0.1267 | f1分数: 88.5815 | time: 307.0052
# Step: 5750 | train loss: 0.0122 | f1分数: 88.8198 | time: 309.6285
# Step: 5800 | train loss: 0.0744 | f1分数: 88.7114 | time: 312.2629
# Step: 5850 | train loss: 0.0262 | f1分数: 89.2479 | time: 314.9160
# Step: 5900 | train loss: 0.0220 | f1分数: 88.4763 | time: 317.5846
# Step: 5950 | train loss: 0.0304 | f1分数: 87.8976 | time: 320.2258
# Step: 6000 | train loss: 0.0434 | f1分数: 88.9052 | time: 322.8740
# Step: 6050 | train loss: 0.0215 | f1分数: 88.3215 | time: 325.5420
# Step: 6100 | train loss: 0.0202 | f1分数: 89.0394 | time: 328.2160
# Step: 6150 | train loss: 0.0293 | f1分数: 88.7104 | time: 330.8179
# Step: 6200 | train loss: 0.0202 | f1分数: 88.9511 | time: 333.4506
# Step: 6250 | train loss: 0.0080 | f1分数: 87.8407 | time: 336.0684
# Step: 6300 | train loss: 0.0301 | f1分数: 88.3096 | time: 338.7233
# Step: 6350 | train loss: 0.0079 | f1分数: 88.7548 | time: 341.3542
# Step: 6400 | train loss: 0.0090 | f1分数: 88.5784 | time: 343.9989
# Step: 6450 | train loss: 0.0276 | f1分数: 89.0499 | time: 346.6374
# Step: 6500 | train loss: 0.0298 | f1分数: 89.2603 | time: 349.2884
# Step: 6550 | train loss: 0.0194 | f1分数: 88.6684 | time: 351.9295
# Step: 6600 | train loss: 0.0087 | f1分数: 89.5097 | time: 354.5580
# Step: 6650 | train loss: 0.0418 | f1分数: 89.6914 | time: 357.2049
# Step: 6700 | train loss: 0.0083 | f1分数: 89.1039 | time: 359.8549
# Step: 6750 | train loss: 0.0083 | f1分数: 89.6484 | time: 362.5213
# Step: 6800 | train loss: 0.0343 | f1分数: 88.5078 | time: 365.1573
# Step: 6850 | train loss: 0.0069 | f1分数: 88.4280 | time: 367.7883
# Step: 6900 | train loss: 0.0167 | f1分数: 88.9285 | time: 370.4369
# Step: 6950 | train loss: 0.0544 | f1分数: 88.7110 | time: 373.0905
# Step: 7000 | train loss: 0.0039 | f1分数: 87.5951 | time: 375.7128
# Step: 7050 | train loss: 0.0105 | f1分数: 89.0267 | time: 378.3731
# Step: 7100 | train loss: 0.0287 | f1分数: 89.2036 | time: 381.0105
# Step: 7150 | train loss: 0.0327 | f1分数: 87.8773 | time: 383.6825
# Step: 7200 | train loss: 0.0067 | f1分数: 88.7575 | time: 386.3254
# Step: 7250 | train loss: 0.0073 | f1分数: 89.1960 | time: 388.9641
# Step: 7300 | train loss: 0.0070 | f1分数: 88.9939 | time: 391.5969
# Step: 7350 | train loss: 0.0247 | f1分数: 89.1254 | time: 394.2295
# Step: 7400 | train loss: 0.0148 | f1分数: 89.5013 | time: 396.8657
# Step: 7450 | train loss: 0.0228 | f1分数: 89.8666 | time: 399.4980
# Step: 7500 | train loss: 0.0015 | f1分数: 89.8263 | time: 402.1232
# Step: 7550 | train loss: 0.0036 | f1分数: 89.7223 | time: 404.8045
# Step: 7600 | train loss: 0.0081 | f1分数: 89.7533 | time: 407.4428
# Step: 7650 | train loss: 0.0021 | f1分数: 89.5435 | time: 410.0459
# Step: 7700 | train loss: 0.0046 | f1分数: 89.3046 | time: 412.7022
# Step: 7750 | train loss: 0.0065 | f1分数: 89.6149 | time: 415.3288
# Step: 7800 | train loss: 0.0331 | f1分数: 89.6018 | time: 418.0004
# Step: 7850 | train loss: 0.0244 | f1分数: 88.9618 | time: 420.6156
# Step: 7900 | train loss: 0.0110 | f1分数: 89.3071 | time: 423.2446
# Step: 7950 | train loss: 0.0084 | f1分数: 89.6669 | time: 425.9073
# Step: 8000 | train loss: 0.0082 | f1分数: 88.6139 | time: 428.5553
# Step: 8050 | train loss: 0.0226 | f1分数: 88.7102 | time: 431.1876
# Step: 8100 | train loss: 0.0039 | f1分数: 88.5229 | time: 433.8151
# Step: 8150 | train loss: 0.0123 | f1分数: 89.3917 | time: 436.4608
# Step: 8200 | train loss: 0.0280 | f1分数: 89.3099 | time: 439.1115
# Step: 8250 | train loss: 0.0076 | f1分数: 89.7526 | time: 441.7610
# Step: 8300 | train loss: 0.0161 | f1分数: 89.2361 | time: 444.3989
# Step: 8350 | train loss: 0.0224 | f1分数: 89.4934 | time: 447.0311
# Step: 8400 | train loss: 0.0032 | f1分数: 89.0840 | time: 449.6847
# Step: 8450 | train loss: 0.0411 | f1分数: 89.0721 | time: 452.3397
# Step: 8500 | train loss: 0.0140 | f1分数: 84.8110 | time: 454.9470
# Step: 8550 | train loss: 0.0487 | f1分数: 88.4655 | time: 457.5783
# Step: 8600 | train loss: 0.0039 | f1分数: 86.9644 | time: 460.2197
# Step: 8650 | train loss: 0.0192 | f1分数: 88.2228 | time: 462.8705
# Step: 8700 | train loss: 0.0007 | f1分数: 88.4390 | time: 465.5039
# Step: 8750 | train loss: 0.0103 | f1分数: 89.3008 | time: 468.1271
# Step: 8800 | train loss: 0.0021 | f1分数: 89.3697 | time: 470.7774
# Step: 8850 | train loss: 0.0055 | f1分数: 89.3736 | time: 473.4246
# Step: 8900 | train loss: 0.0009 | f1分数: 89.0269 | time: 476.0557
# Step: 8950 | train loss: 0.0098 | f1分数: 89.4232 | time: 478.7208
# Step: 9000 | train loss: 0.0009 | f1分数: 89.4454 | time: 481.3778
# Step: 9050 | train loss: 0.0013 | f1分数: 89.3310 | time: 484.0244
# Step: 9100 | train loss: 0.0286 | f1分数: 89.3747 | time: 486.7078
# Step: 9150 | train loss: 0.0108 | f1分数: 89.4231 | time: 489.3629
# Step: 9200 | train loss: 0.0011 | f1分数: 89.6111 | time: 492.0062
# Step: 9250 | train loss: 0.0043 | f1分数: 89.6095 | time: 494.6248
# Step: 9300 | train loss: 0.0426 | f1分数: 88.7188 | time: 497.2625
# Step: 9350 | train loss: 0.0035 | f1分数: 89.7230 | time: 499.8921
# Step: 9400 | train loss: 0.0036 | f1分数: 89.2535 | time: 502.5117
# Step: 9450 | train loss: 0.0164 | f1分数: 89.0959 | time: 505.1702
# Step: 9500 | train loss: 0.0008 | f1分数: 88.8890 | time: 507.8009
# Step: 9550 | train loss: 0.0026 | f1分数: 89.1839 | time: 510.4652
# Step: 9600 | train loss: 0.0146 | f1分数: 89.1486 | time: 513.1362
# Step: 9650 | train loss: 0.0290 | f1分数: 89.0617 | time: 515.8126
# Step: 9700 | train loss: 0.0086 | f1分数: 88.8992 | time: 518.4554
# Step: 9750 | train loss: 0.0039 | f1分数: 88.2715 | time: 521.1232
# Step: 9800 | train loss: 0.0095 | f1分数: 89.0107 | time: 523.7385
# Step: 9850 | train loss: 0.0029 | f1分数: 87.9850 | time: 526.3956
# Step: 9900 | train loss: 0.0037 | f1分数: 88.9849 | time: 529.0823
# Step: 9950 | train loss: 0.0146 | f1分数: 89.4086 | time: 531.7039
# Step: 10000 | train loss: 0.0052 | f1分数: 89.1099 | time: 534.3579
# Step: 10050 | train loss: 0.0046 | f1分数: 88.3236 | time: 536.9856
# Step: 10100 | train loss: 0.0026 | f1分数: 88.7594 | time: 539.6022
# Step: 10150 | train loss: 0.0007 | f1分数: 89.1644 | time: 542.2280
# Step: 10200 | train loss: 0.0012 | f1分数: 89.2823 | time: 544.8868
# Step: 10250 | train loss: 0.0037 | f1分数: 89.2325 | time: 547.5141
# Step: 10300 | train loss: 0.0004 | f1分数: 89.3962 | time: 550.1448
# Step: 10350 | train loss: 0.0134 | f1分数: 89.2860 | time: 552.7726
# Step: 10400 | train loss: 0.0057 | f1分数: 89.3448 | time: 555.4133
# Step: 10450 | train loss: 0.0059 | f1分数: 89.2025 | time: 558.0564
# Step: 10500 | train loss: 0.0011 | f1分数: 89.6564 | time: 560.7048
# Step: 10550 | train loss: 0.0004 | f1分数: 89.5420 | time: 563.3027
# Step: 10600 | train loss: 0.0005 | f1分数: 89.2830 | time: 565.8894
# Step: 10650 | train loss: 0.0063 | f1分数: 87.7526 | time: 568.5247
# Step: 10700 | train loss: 0.0157 | f1分数: 88.0404 | time: 571.1350
# Step: 10750 | train loss: 0.0068 | f1分数: 88.4910 | time: 573.7762
# Step: 10800 | train loss: 0.0423 | f1分数: 89.1528 | time: 576.4193
# Step: 10850 | train loss: 0.0411 | f1分数: 85.5755 | time: 579.0766
# Step: 10900 | train loss: 0.0168 | f1分数: 88.6164 | time: 581.7168
# Step: 10950 | train loss: 0.0128 | f1分数: 88.7708 | time: 584.3563
# Step: 11000 | train loss: 0.0225 | f1分数: 89.2348 | time: 586.9815
# Step: 11050 | train loss: 0.0056 | f1分数: 88.8482 | time: 589.6003
# Step: 11100 | train loss: 0.0053 | f1分数: 89.4130 | time: 592.2550
# Step: 11150 | train loss: 0.0089 | f1分数: 89.4064 | time: 594.9373
# Step: 11200 | train loss: 0.0160 | f1分数: 89.1693 | time: 597.5960
# Step: 11250 | train loss: 0.0024 | f1分数: 88.5205 | time: 600.2275
# Step: 11300 | train loss: 0.0067 | f1分数: 89.4365 | time: 602.8248
# Step: 11350 | train loss: 0.0011 | f1分数: 89.3574 | time: 605.4908
# Step: 11400 | train loss: 0.0003 | f1分数: 89.2120 | time: 608.1099
# Step: 11450 | train loss: 0.0033 | f1分数: 89.4234 | time: 610.7335
# Step: 11500 | train loss: 0.0009 | f1分数: 89.5480 | time: 613.3797
# Step: 11550 | train loss: 0.0016 | f1分数: 89.3749 | time: 616.0061
# Step: 11600 | train loss: 0.0010 | f1分数: 89.5742 | time: 618.6147
# Step: 11650 | train loss: 0.0004 | f1分数: 89.0344 | time: 621.2371
# Step: 11700 | train loss: 0.0039 | f1分数: 89.6156 | time: 623.8977
# Step: 11750 | train loss: 0.0041 | f1分数: 89.6226 | time: 626.5471
# Step: 11800 | train loss: 0.0094 | f1分数: 88.7358 | time: 629.1761
# Step: 11850 | train loss: 0.0018 | f1分数: 88.7777 | time: 631.8255
# Step: 11900 | train loss: 0.0011 | f1分数: 89.3522 | time: 634.4729
# Step: 11950 | train loss: 0.0006 | f1分数: 89.6759 | time: 637.0872
# Step: 12000 | train loss: 0.0007 | f1分数: 89.7292 | time: 639.7241
# Step: 12050 | train loss: 0.0005 | f1分数: 89.7019 | time: 642.3583
# Step: 12100 | train loss: 0.0351 | f1分数: 89.8715 | time: 644.9835
# Step: 12150 | train loss: 0.0002 | f1分数: 89.5298 | time: 647.6096
# Step: 12200 | train loss: 0.0009 | f1分数: 89.6820 | time: 650.2944
# Step: 12250 | train loss: 0.0023 | f1分数: 89.4736 | time: 652.9377
# Step: 12300 | train loss: 0.0006 | f1分数: 89.0851 | time: 655.5753
# Step: 12350 | train loss: 0.0212 | f1分数: 88.9149 | time: 658.2267
# Step: 12400 | train loss: 0.0099 | f1分数: 88.6709 | time: 660.8499
# Step: 12450 | train loss: 0.0188 | f1分数: 88.4205 | time: 663.5063
# Step: 12500 | train loss: 0.0115 | f1分数: 89.0874 | time: 666.1568
# Step: 12550 | train loss: 0.0098 | f1分数: 88.4903 | time: 668.8127
# Step: 12600 | train loss: 0.0125 | f1分数: 89.7914 | time: 671.4635
# Step: 12650 | train loss: 0.0046 | f1分数: 89.7502 | time: 674.0767
# Step: 12700 | train loss: 0.0004 | f1分数: 89.6694 | time: 676.7078
# Step: 12750 | train loss: 0.0028 | f1分数: 89.5690 | time: 679.3521
# Step: 12800 | train loss: 0.0007 | f1分数: 89.8309 | time: 682.0002
# Step: 12850 | train loss: 0.0022 | f1分数: 89.5843 | time: 684.6330
# Step: 12900 | train loss: 0.0007 | f1分数: 89.8590 | time: 687.3016
# Step: 12950 | train loss: 0.0002 | f1分数: 89.8321 | time: 689.9530
# Step: 13000 | train loss: 0.0045 | f1分数: 89.9670 | time: 692.5917
# Step: 13050 | train loss: 0.0041 | f1分数: 89.7801 | time: 695.2496
# Step: 13100 | train loss: 0.0010 | f1分数: 89.9499 | time: 697.8766
# Step: 13150 | train loss: 0.0004 | f1分数: 89.8884 | time: 700.4978
# Step: 13200 | train loss: 0.0002 | f1分数: 89.8446 | time: 703.1351
# Step: 13250 | train loss: 0.0002 | f1分数: 89.9205 | time: 705.7755
# Step: 13300 | train loss: 0.0002 | f1分数: 89.8238 | time: 708.4289
# Step: 13350 | train loss: 0.0007 | f1分数: 89.8447 | time: 711.0517
# Step: 13400 | train loss: 0.0001 | f1分数: 89.8611 | time: 713.7163
# Step: 13450 | train loss: 0.0047 | f1分数: 89.8540 | time: 716.3432
# Step: 13500 | train loss: 0.0037 | f1分数: 89.8729 | time: 718.9711
# Step: 13550 | train loss: 0.0001 | f1分数: 89.9729 | time: 721.5892
# Step: 13600 | train loss: 0.0006 | f1分数: 89.3023 | time: 724.2219
# Step: 13650 | train loss: 0.0060 | f1分数: 89.8439 | time: 726.8333
# Step: 13700 | train loss: 0.0002 | f1分数: 89.1721 | time: 729.4797
# Step: 13750 | train loss: 0.0035 | f1分数: 88.5788 | time: 732.0910
# Step: 13800 | train loss: 0.0282 | f1分数: 83.3756 | time: 734.7302
# Step: 13850 | train loss: 0.0341 | f1分数: 88.3360 | time: 737.4056
# Step: 13900 | train loss: 0.0341 | f1分数: 88.5933 | time: 740.0292

# kernel_size = (1,8),batch_norm
# Step: 50 | train loss: 1.7263 | f1分数: 26.5857 | time: 5.9092
# Step: 100 | train loss: 0.4173 | f1分数: 77.0701 | time: 10.3989
# Step: 150 | train loss: 0.4496 | f1分数: 80.7084 | time: 14.8931
# Step: 200 | train loss: 0.3686 | f1分数: 82.9937 | time: 19.4226
# Step: 250 | train loss: 0.3401 | f1分数: 85.5486 | time: 23.9438
# Step: 300 | train loss: 0.6819 | f1分数: 85.7884 | time: 28.4302
# Step: 350 | train loss: 0.1377 | f1分数: 82.0294 | time: 32.9188
# Step: 400 | train loss: 0.6961 | f1分数: 79.8344 | time: 37.4721
# Step: 450 | train loss: 0.2928 | f1分数: 87.8218 | time: 42.0251
# Step: 500 | train loss: 0.2673 | f1分数: 87.7110 | time: 46.5624
# Step: 550 | train loss: 0.2239 | f1分数: 85.1145 | time: 51.0768
# Step: 600 | train loss: 0.1507 | f1分数: 86.7971 | time: 55.6293
# Step: 650 | train loss: 0.1622 | f1分数: 87.1951 | time: 60.1636
# Step: 700 | train loss: 0.2519 | f1分数: 81.1827 | time: 64.6787
# Step: 750 | train loss: 0.1266 | f1分数: 83.0810 | time: 69.1798
# Step: 800 | train loss: 0.1250 | f1分数: 84.3584 | time: 73.6748
# Step: 850 | train loss: 0.1656 | f1分数: 86.9023 | time: 78.2153
# Step: 900 | train loss: 0.2200 | f1分数: 89.1554 | time: 82.7487
# Step: 950 | train loss: 0.1282 | f1分数: 88.7106 | time: 87.2702
# Step: 1000 | train loss: 0.1172 | f1分数: 86.6977 | time: 91.7794
# Step: 1050 | train loss: 0.0990 | f1分数: 85.9038 | time: 96.2920
# Step: 1100 | train loss: 0.0810 | f1分数: 87.8423 | time: 100.8079
# Step: 1150 | train loss: 0.2274 | f1分数: 85.8558 | time: 105.3252
# Step: 1200 | train loss: 0.0979 | f1分数: 86.7050 | time: 109.8471
# Step: 1250 | train loss: 0.1487 | f1分数: 89.1148 | time: 114.4087
# Step: 1300 | train loss: 0.0639 | f1分数: 87.8503 | time: 118.9492
# Step: 1350 | train loss: 0.1558 | f1分数: 89.7807 | time: 123.4697
# Step: 1400 | train loss: 0.0123 | f1分数: 86.0963 | time: 128.0179
# Step: 1450 | train loss: 0.0739 | f1分数: 86.6928 | time: 132.5514
# Step: 1500 | train loss: 0.0402 | f1分数: 88.2484 | time: 137.0800
# Step: 1550 | train loss: 0.0233 | f1分数: 88.2177 | time: 141.6541
# Step: 1600 | train loss: 0.0432 | f1分数: 89.2679 | time: 146.1547
# Step: 1650 | train loss: 0.0744 | f1分数: 88.9127 | time: 150.7274
# Step: 1700 | train loss: 0.0381 | f1分数: 87.0325 | time: 155.2824
# Step: 1750 | train loss: 0.0343 | f1分数: 86.8998 | time: 159.8275
# Step: 1800 | train loss: 0.0480 | f1分数: 90.5580 | time: 164.3782
# Step: 1850 | train loss: 0.0443 | f1分数: 75.1186 | time: 168.9301
# Step: 1900 | train loss: 0.0222 | f1分数: 88.7301 | time: 173.4940
# Step: 1950 | train loss: 0.0528 | f1分数: 84.6852 | time: 178.0157
# Step: 2000 | train loss: 0.1191 | f1分数: 88.1195 | time: 182.5293
# Step: 2050 | train loss: 0.0353 | f1分数: 90.0663 | time: 187.1028
# Step: 2100 | train loss: 0.0735 | f1分数: 88.8409 | time: 191.6315
# Step: 2150 | train loss: 0.1244 | f1分数: 86.8524 | time: 196.1828
# Step: 2200 | train loss: 0.0181 | f1分数: 90.2025 | time: 200.6968
# Step: 2250 | train loss: 0.0760 | f1分数: 88.3154 | time: 205.2505
# Step: 2300 | train loss: 0.0527 | f1分数: 87.8340 | time: 209.8067
# Step: 2350 | train loss: 0.0085 | f1分数: 88.1986 | time: 214.3661
# Step: 2400 | train loss: 0.0174 | f1分数: 88.5636 | time: 218.8773
# Step: 2450 | train loss: 0.0046 | f1分数: 89.7875 | time: 223.4349
# Step: 2500 | train loss: 0.0490 | f1分数: 89.9630 | time: 227.9575
# Step: 2550 | train loss: 0.0108 | f1分数: 89.8679 | time: 232.4792
# Step: 2600 | train loss: 0.0085 | f1分数: 89.0681 | time: 237.0323
# Step: 2650 | train loss: 0.0175 | f1分数: 89.8476 | time: 241.6038
# Step: 2700 | train loss: 0.0203 | f1分数: 89.7479 | time: 246.1424
# Step: 2750 | train loss: 0.0048 | f1分数: 90.4105 | time: 250.7259
# Step: 2800 | train loss: 0.0130 | f1分数: 89.3195 | time: 255.2772
# Step: 2850 | train loss: 0.0049 | f1分数: 88.2885 | time: 259.8354
# Step: 2900 | train loss: 0.0128 | f1分数: 88.5975 | time: 264.3809
# Step: 2950 | train loss: 0.0725 | f1分数: 88.4474 | time: 268.9091
# Step: 3000 | train loss: 0.0089 | f1分数: 87.9339 | time: 273.4583
# Step: 3050 | train loss: 0.1260 | f1分数: 87.8474 | time: 277.9972
# Step: 3100 | train loss: 0.0341 | f1分数: 89.1009 | time: 282.5552
# Step: 3150 | train loss: 0.0334 | f1分数: 89.0204 | time: 287.0989
# Step: 3200 | train loss: 0.0221 | f1分数: 87.1752 | time: 291.6517
# Step: 3250 | train loss: 0.0193 | f1分数: 89.0797 | time: 296.2223
# Step: 3300 | train loss: 0.0184 | f1分数: 86.5049 | time: 300.7775
# Step: 3350 | train loss: 0.0317 | f1分数: 88.2885 | time: 305.3557
# Step: 3400 | train loss: 0.0242 | f1分数: 90.0787 | time: 309.9296
# Step: 3450 | train loss: 0.0194 | f1分数: 89.3418 | time: 314.4839
# Step: 3500 | train loss: 0.0237 | f1分数: 89.6564 | time: 319.0408
# Step: 3550 | train loss: 0.0163 | f1分数: 89.8993 | time: 323.5765
# Step: 3600 | train loss: 0.0173 | f1分数: 89.9063 | time: 328.0360
# Step: 3650 | train loss: 0.0111 | f1分数: 88.8116 | time: 332.5395
# Step: 3700 | train loss: 0.0015 | f1分数: 90.0150 | time: 337.0973
# Step: 3750 | train loss: 0.0050 | f1分数: 88.9838 | time: 341.6907
# Step: 3800 | train loss: 0.0634 | f1分数: 90.0747 | time: 346.2639
# Step: 3850 | train loss: 0.0186 | f1分数: 89.9283 | time: 350.8175
# Step: 3900 | train loss: 0.0239 | f1分数: 89.3529 | time: 355.3797
# Step: 3950 | train loss: 0.0259 | f1分数: 87.3330 | time: 359.8995
# Step: 4000 | train loss: 0.0212 | f1分数: 90.2060 | time: 364.4415
# Step: 4050 | train loss: 0.0060 | f1分数: 86.3566 | time: 368.9773
# Step: 4100 | train loss: 0.0313 | f1分数: 88.2935 | time: 373.4896
# Step: 4150 | train loss: 0.0118 | f1分数: 82.3168 | time: 378.0281
# Step: 4200 | train loss: 0.0082 | f1分数: 85.8376 | time: 382.5934
# Step: 4250 | train loss: 0.0109 | f1分数: 88.1274 | time: 387.1639
# Step: 4300 | train loss: 0.0359 | f1分数: 89.3356 | time: 391.7107
# Step: 4350 | train loss: 0.0020 | f1分数: 88.9787 | time: 396.2563
# Step: 4400 | train loss: 0.0145 | f1分数: 89.5056 | time: 400.8707
# Step: 4450 | train loss: 0.0041 | f1分数: 88.0168 | time: 405.4198
# Step: 4500 | train loss: 0.0155 | f1分数: 89.1700 | time: 409.9633
# Step: 4550 | train loss: 0.0087 | f1分数: 89.7280 | time: 414.5111
# Step: 4600 | train loss: 0.0032 | f1分数: 89.9038 | time: 419.0356
# Step: 4650 | train loss: 0.0032 | f1分数: 90.4106 | time: 423.6041
# Step: 4700 | train loss: 0.0029 | f1分数: 89.9286 | time: 428.1308
# Step: 4750 | train loss: 0.0005 | f1分数: 90.0051 | time: 432.6988
# Step: 4800 | train loss: 0.0020 | f1分数: 90.3957 | time: 437.2286
# Step: 4850 | train loss: 0.0068 | f1分数: 89.7952 | time: 441.7373
# Step: 4900 | train loss: 0.0013 | f1分数: 89.8709 | time: 446.2906
# Step: 4950 | train loss: 0.0432 | f1分数: 89.5903 | time: 450.8358
# Step: 5000 | train loss: 0.0003 | f1分数: 90.0850 | time: 455.4217
# Step: 5050 | train loss: 0.0015 | f1分数: 90.5037 | time: 459.9704
# Step: 5100 | train loss: 0.0037 | f1分数: 89.5040 | time: 464.4951
# Step: 5150 | train loss: 0.0036 | f1分数: 90.0613 | time: 469.0805
# Step: 5200 | train loss: 0.0005 | f1分数: 90.0273 | time: 473.6608
# Step: 5250 | train loss: 0.0002 | f1分数: 90.3867 | time: 478.1263
# Step: 5300 | train loss: 0.0003 | f1分数: 90.4416 | time: 482.7262
# Step: 5350 | train loss: 0.0004 | f1分数: 90.3532 | time: 487.2918
# Step: 5400 | train loss: 0.0008 | f1分数: 90.4001 | time: 491.8882
# Step: 5450 | train loss: 0.0002 | f1分数: 90.4441 | time: 496.3906
# Step: 5500 | train loss: 0.0088 | f1分数: 90.3930 | time: 500.8524
# Step: 5550 | train loss: 0.0007 | f1分数: 90.3387 | time: 505.3455
# Step: 5600 | train loss: 0.0002 | f1分数: 90.3817 | time: 509.8552
# Step: 5650 | train loss: 0.0002 | f1分数: 90.3522 | time: 514.3977
# Step: 5700 | train loss: 0.0054 | f1分数: 90.4868 | time: 518.9803
# Step: 5750 | train loss: 0.0001 | f1分数: 90.4798 | time: 523.5236
# Step: 5800 | train loss: 0.0002 | f1分数: 90.5047 | time: 528.0582
# Step: 5850 | train loss: 0.0003 | f1分数: 90.3772 | time: 532.6294
# Step: 5900 | train loss: 0.0001 | f1分数: 90.3992 | time: 537.1798
# Step: 5950 | train loss: 0.0001 | f1分数: 90.3105 | time: 541.7424
# Step: 6000 | train loss: 0.0039 | f1分数: 88.7908 | time: 546.2746
# Step: 6050 | train loss: 0.0026 | f1分数: 89.5012 | time: 550.8710
# Step: 6100 | train loss: 0.0029 | f1分数: 88.2843 | time: 555.4032
# Step: 6150 | train loss: 0.0257 | f1分数: 86.9268 | time: 559.8603
# Step: 6200 | train loss: 0.1085 | f1分数: 87.1006 | time: 564.2764
# Step: 6250 | train loss: 0.0486 | f1分数: 84.0352 | time: 568.7651
# Step: 6300 | train loss: 0.0654 | f1分数: 88.9288 | time: 573.1973
# Step: 6350 | train loss: 0.0148 | f1分数: 89.0358 | time: 577.6689
# Step: 6400 | train loss: 0.0131 | f1分数: 87.7597 | time: 582.1450
# Step: 6450 | train loss: 0.0400 | f1分数: 88.9277 | time: 586.6749
# Step: 6500 | train loss: 0.0169 | f1分数: 89.9444 | time: 591.2332
# Step: 6550 | train loss: 0.0144 | f1分数: 89.8164 | time: 595.7302
# Step: 6600 | train loss: 0.0017 | f1分数: 89.7484 | time: 600.2830
# Step: 6650 | train loss: 0.0277 | f1分数: 89.7980 | time: 604.8449
# Step: 6700 | train loss: 0.0056 | f1分数: 88.5021 | time: 609.3838
# Step: 6750 | train loss: 0.0050 | f1分数: 87.3465 | time: 613.8828
# Step: 6800 | train loss: 0.0017 | f1分数: 90.1074 | time: 618.3661
# Step: 6850 | train loss: 0.0005 | f1分数: 89.9063 | time: 622.9320
# Step: 6900 | train loss: 0.0014 | f1分数: 90.0925 | time: 627.4733
# Step: 6950 | train loss: 0.0088 | f1分数: 90.0024 | time: 632.0073
# Step: 7000 | train loss: 0.0003 | f1分数: 89.8424 | time: 636.5094
# Step: 7050 | train loss: 0.0132 | f1分数: 89.6719 | time: 641.0395
# Step: 7100 | train loss: 0.0012 | f1分数: 89.5755 | time: 645.5207
# Step: 7150 | train loss: 0.0244 | f1分数: 89.9143 | time: 650.0387
# Step: 7200 | train loss: 0.0035 | f1分数: 89.1829 | time: 654.5665
# Step: 7250 | train loss: 0.0011 | f1分数: 89.4564 | time: 659.1082
# Step: 7300 | train loss: 0.0035 | f1分数: 90.0467 | time: 663.6172
# Step: 7350 | train loss: 0.0013 | f1分数: 89.0146 | time: 668.1036
# Step: 7400 | train loss: 0.0004 | f1分数: 90.0481 | time: 672.6068
# Step: 7450 | train loss: 0.0005 | f1分数: 90.0186 | time: 677.1413
# Step: 7500 | train loss: 0.0002 | f1分数: 89.7770 | time: 681.7052
# Step: 7550 | train loss: 0.0005 | f1分数: 89.5462 | time: 686.2227
# Step: 7600 | train loss: 0.0058 | f1分数: 89.9499 | time: 690.7591
# Step: 7650 | train loss: 0.0003 | f1分数: 89.8587 | time: 695.2286
# Step: 7700 | train loss: 0.0003 | f1分数: 89.9610 | time: 699.7432
# Step: 7750 | train loss: 0.0042 | f1分数: 89.8267 | time: 704.2624
# Step: 7800 | train loss: 0.0066 | f1分数: 89.8813 | time: 708.8358
# Step: 7850 | train loss: 0.0016 | f1分数: 90.0162 | time: 713.3945
# Step: 7900 | train loss: 0.0001 | f1分数: 90.1025 | time: 717.9445
# Step: 7950 | train loss: 0.0002 | f1分数: 90.0506 | time: 722.5056
# Step: 8000 | train loss: 0.0002 | f1分数: 90.2328 | time: 727.0315
# Step: 8050 | train loss: 0.0005 | f1分数: 89.9948 | time: 731.5632
# Step: 8100 | train loss: 0.0001 | f1分数: 90.0836 | time: 736.0937
# Step: 8150 | train loss: 0.0103 | f1分数: 90.0671 | time: 740.6179
# Step: 8200 | train loss: 0.0003 | f1分数: 90.0096 | time: 745.1857
# Step: 8250 | train loss: 0.0001 | f1分数: 89.9650 | time: 749.7290
# Step: 8300 | train loss: 0.0001 | f1分数: 90.1573 | time: 754.2563
# Step: 8350 | train loss: 0.0004 | f1分数: 90.1196 | time: 758.7794
# Step: 8400 | train loss: 0.0001 | f1分数: 90.0394 | time: 763.2484
# Step: 8450 | train loss: 0.0001 | f1分数: 90.1420 | time: 767.7103
# Step: 8500 | train loss: 0.0002 | f1分数: 90.1955 | time: 772.2346
# Step: 8550 | train loss: 0.0003 | f1分数: 90.2475 | time: 776.7306
# Step: 8600 | train loss: 0.0000 | f1分数: 90.2802 | time: 781.2657
# Step: 8650 | train loss: 0.0001 | f1分数: 90.1364 | time: 785.7876
# Step: 8700 | train loss: 0.0000 | f1分数: 90.1482 | time: 790.3571
# Step: 8750 | train loss: 0.0000 | f1分数: 90.2669 | time: 794.9366
# Step: 8800 | train loss: 0.0001 | f1分数: 90.2536 | time: 799.4362
# Step: 8850 | train loss: 0.0003 | f1分数: 90.2204 | time: 803.9790
# Step: 8900 | train loss: 0.0001 | f1分数: 89.4630 | time: 808.5435
# Step: 8950 | train loss: 0.0009 | f1分数: 88.4364 | time: 813.1319
# Step: 9000 | train loss: 0.0059 | f1分数: 87.5366 | time: 817.6956
# Step: 9050 | train loss: 0.2188 | f1分数: 86.4580 | time: 822.2376
# Step: 9100 | train loss: 0.1064 | f1分数: 88.1313 | time: 826.8067
# Step: 9150 | train loss: 0.0499 | f1分数: 89.1756 | time: 831.3867
# Step: 9200 | train loss: 0.0920 | f1分数: 86.1856 | time: 835.9225
# Step: 9250 | train loss: 0.0099 | f1分数: 89.3057 | time: 840.4480
# Step: 9300 | train loss: 0.0415 | f1分数: 89.5349 | time: 844.9543
# Step: 9350 | train loss: 0.0181 | f1分数: 89.2782 | time: 849.4796
# Step: 9400 | train loss: 0.0028 | f1分数: 89.5665 | time: 854.0221
# Step: 9450 | train loss: 0.0232 | f1分数: 88.6175 | time: 858.5280
# Step: 9500 | train loss: 0.0018 | f1分数: 89.5990 | time: 863.0653
# Step: 9550 | train loss: 0.0016 | f1分数: 88.2740 | time: 867.5823
# Step: 9600 | train loss: 0.0147 | f1分数: 89.3333 | time: 872.1268
# Step: 9650 | train loss: 0.0023 | f1分数: 90.2466 | time: 876.6558
# Step: 9700 | train loss: 0.0031 | f1分数: 89.8879 | time: 881.1824
# Step: 9750 | train loss: 0.0006 | f1分数: 88.1422 | time: 885.7226
# Step: 9800 | train loss: 0.0198 | f1分数: 87.5231 | time: 890.2893
# Step: 9850 | train loss: 0.0020 | f1分数: 89.3983 | time: 894.8177
# Step: 9900 | train loss: 0.0017 | f1分数: 89.8201 | time: 899.3671
# Step: 9950 | train loss: 0.0009 | f1分数: 90.1368 | time: 903.9118
# Step: 10000 | train loss: 0.0074 | f1分数: 89.2598 | time: 908.4157
# Step: 10050 | train loss: 0.0004 | f1分数: 90.0129 | time: 912.9173
# Step: 10100 | train loss: 0.0007 | f1分数: 89.7435 | time: 917.4903
# Step: 10150 | train loss: 0.0006 | f1分数: 89.8941 | time: 922.0185
# Step: 10200 | train loss: 0.0022 | f1分数: 90.2785 | time: 926.5491
# Step: 10250 | train loss: 0.0026 | f1分数: 90.2692 | time: 930.9983
# Step: 10300 | train loss: 0.0001 | f1分数: 90.1706 | time: 935.5197
# Step: 10350 | train loss: 0.0019 | f1分数: 90.1021 | time: 940.0179
# Step: 10400 | train loss: 0.0036 | f1分数: 90.1986 | time: 944.5150
# Step: 10450 | train loss: 0.0002 | f1分数: 90.3217 | time: 949.0489
# Step: 10500 | train loss: 0.0003 | f1分数: 90.2638 | time: 953.5794
# Step: 10550 | train loss: 0.0000 | f1分数: 90.4986 | time: 958.1316
# Step: 10600 | train loss: 0.0001 | f1分数: 90.3863 | time: 962.6556
# Step: 10650 | train loss: 0.0003 | f1分数: 90.4719 | time: 967.1667
# Step: 10700 | train loss: 0.0018 | f1分数: 90.2799 | time: 971.6199
# Step: 10750 | train loss: 0.0001 | f1分数: 89.9674 | time: 976.0778
# Step: 10800 | train loss: 0.0109 | f1分数: 90.6606 | time: 980.5713
# Step: 10850 | train loss: 0.0003 | f1分数: 90.4429 | time: 985.1341
# Step: 10900 | train loss: 0.0001 | f1分数: 90.3805 | time: 989.7009
# Step: 10950 | train loss: 0.0004 | f1分数: 90.2692 | time: 994.2656
# Step: 11000 | train loss: 0.2658 | f1分数: 88.9239 | time: 998.7561
# Step: 11050 | train loss: 0.0057 | f1分数: 87.9884 | time: 1003.2139
# Step: 11100 | train loss: 0.0395 | f1分数: 87.3741 | time: 1007.7310
# Step: 11150 | train loss: 0.0181 | f1分数: 88.0624 | time: 1012.2832
# Step: 11200 | train loss: 0.0443 | f1分数: 89.3502 | time: 1016.8497
# Step: 11250 | train loss: 0.0079 | f1分数: 89.7325 | time: 1021.4362
# Step: 11300 | train loss: 0.0273 | f1分数: 89.6988 | time: 1026.0225
# Step: 11350 | train loss: 0.0020 | f1分数: 88.9245 | time: 1030.5945
# Step: 11400 | train loss: 0.0080 | f1分数: 89.3712 | time: 1035.1513
# Step: 11450 | train loss: 0.0007 | f1分数: 89.5084 | time: 1039.7164
# Step: 11500 | train loss: 0.0009 | f1分数: 89.6420 | time: 1044.1947
# Step: 11550 | train loss: 0.0032 | f1分数: 90.0198 | time: 1048.5869
# Step: 11600 | train loss: 0.0004 | f1分数: 90.2546 | time: 1053.0253
# Step: 11650 | train loss: 0.0003 | f1分数: 89.9314 | time: 1057.4739
# Step: 11700 | train loss: 0.0003 | f1分数: 88.9740 | time: 1061.8892
# Step: 11750 | train loss: 0.0006 | f1分数: 89.7919 | time: 1066.3105
# Step: 11800 | train loss: 0.0004 | f1分数: 90.0169 | time: 1070.8589
# Step: 11850 | train loss: 0.0001 | f1分数: 89.9067 | time: 1075.3831
# Step: 11900 | train loss: 0.0001 | f1分数: 90.0872 | time: 1079.8778
# Step: 11950 | train loss: 0.0002 | f1分数: 90.1774 | time: 1084.4112
# Step: 12000 | train loss: 0.0002 | f1分数: 90.1935 | time: 1088.9448
# Step: 12050 | train loss: 0.0001 | f1分数: 89.9571 | time: 1093.5185
# Step: 12100 | train loss: 0.0002 | f1分数: 90.0424 | time: 1098.0190
# Step: 12150 | train loss: 0.0001 | f1分数: 90.0178 | time: 1102.4669
# Step: 12200 | train loss: 0.0001 | f1分数: 90.0271 | time: 1106.8132
# Step: 12250 | train loss: 0.0001 | f1分数: 90.0187 | time: 1111.2583
# Step: 12300 | train loss: 0.0001 | f1分数: 89.9999 | time: 1115.6554
# Step: 12350 | train loss: 0.0001 | f1分数: 89.9822 | time: 1120.1139
# Step: 12400 | train loss: 0.0000 | f1分数: 89.8829 | time: 1124.5684
# Step: 12450 | train loss: 0.0001 | f1分数: 89.9932 | time: 1129.0708
# Step: 12500 | train loss: 0.0001 | f1分数: 90.0841 | time: 1133.5968
# Step: 12550 | train loss: 0.0000 | f1分数: 90.0864 | time: 1138.1306
# Step: 12600 | train loss: 0.0001 | f1分数: 90.1043 | time: 1142.6486
# Step: 12650 | train loss: 0.0001 | f1分数: 90.0168 | time: 1147.2025
# Step: 12700 | train loss: 0.0000 | f1分数: 90.0570 | time: 1151.5897
# Step: 12750 | train loss: 0.0000 | f1分数: 90.0191 | time: 1156.0861
# Step: 12800 | train loss: 0.0000 | f1分数: 89.9389 | time: 1160.5793
# Step: 12850 | train loss: 0.0001 | f1分数: 89.8601 | time: 1165.0397
# Step: 12900 | train loss: 0.0001 | f1分数: 90.2072 | time: 1169.5030
# Step: 12950 | train loss: 0.0001 | f1分数: 89.8072 | time: 1174.0018
# Step: 13000 | train loss: 0.0007 | f1分数: 89.5540 | time: 1178.4911
# Step: 13050 | train loss: 0.0102 | f1分数: 87.5649 | time: 1183.0064
# Step: 13100 | train loss: 0.0261 | f1分数: 83.5857 | time: 1187.4443
# Step: 13150 | train loss: 0.0443 | f1分数: 87.5896 | time: 1191.8808
# Step: 13200 | train loss: 0.0372 | f1分数: 84.1037 | time: 1196.3286
# Step: 13250 | train loss: 0.0318 | f1分数: 89.1263 | time: 1200.8637
# Step: 13300 | train loss: 0.0413 | f1分数: 89.5103 | time: 1205.4048
# Step: 13350 | train loss: 0.0330 | f1分数: 89.8184 | time: 1209.9884
# Step: 13400 | train loss: 0.0012 | f1分数: 89.8201 | time: 1214.5038
# Step: 13450 | train loss: 0.0078 | f1分数: 89.7224 | time: 1218.9977
# Step: 13500 | train loss: 0.0068 | f1分数: 89.8319 | time: 1223.5461
# Step: 13550 | train loss: 0.0105 | f1分数: 89.5039 | time: 1228.0919
# Step: 13600 | train loss: 0.0089 | f1分数: 89.8048 | time: 1232.6353
# Step: 13650 | train loss: 0.0036 | f1分数: 83.3723 | time: 1237.1300
# Step: 13700 | train loss: 0.0166 | f1分数: 89.2785 | time: 1241.5838
# Step: 13750 | train loss: 0.0495 | f1分数: 89.1338 | time: 1246.0558
# Step: 13800 | train loss: 0.0122 | f1分数: 88.9879 | time: 1250.5756
# Step: 13850 | train loss: 0.0047 | f1分数: 89.6705 | time: 1255.1023
# Step: 13900 | train loss: 0.0003 | f1分数: 89.7908 | time: 1259.6129

# padding_size = (2, 1)
# Step: 50 | train loss: 1.7227 | f1分数: 75.4185 | time: 8.8132
# Step: 100 | train loss: 0.4072 | f1分数: 75.6818 | time: 16.0450
# Step: 150 | train loss: 0.5194 | f1分数: 80.3115 | time: 23.3118
# Step: 200 | train loss: 0.4589 | f1分数: 79.9756 | time: 30.5850
# Step: 250 | train loss: 0.5235 | f1分数: 82.2186 | time: 38.0302
# Step: 300 | train loss: 0.8468 | f1分数: 85.6147 | time: 45.4261
# Step: 350 | train loss: 0.1528 | f1分数: 85.9038 | time: 52.8115
# Step: 400 | train loss: 0.5235 | f1分数: 85.2041 | time: 60.2382
# Step: 450 | train loss: 0.3473 | f1分数: 86.1653 | time: 67.6163
# Step: 500 | train loss: 0.4319 | f1分数: 87.5595 | time: 74.9406
# Step: 550 | train loss: 0.4031 | f1分数: 84.0883 | time: 82.2631
# Step: 600 | train loss: 0.1946 | f1分数: 85.5718 | time: 89.6961
# Step: 650 | train loss: 0.2995 | f1分数: 82.8574 | time: 96.9590
# Step: 700 | train loss: 0.3166 | f1分数: 83.0574 | time: 104.2037
# Step: 750 | train loss: 0.2111 | f1分数: 83.2630 | time: 111.5625
# Step: 800 | train loss: 0.2065 | f1分数: 82.7171 | time: 118.9701
# Step: 850 | train loss: 0.2702 | f1分数: 84.3476 | time: 126.3707
# Step: 900 | train loss: 0.4470 | f1分数: 84.5746 | time: 133.5984
# Step: 950 | train loss: 0.2798 | f1分数: 85.2875 | time: 140.8778
# Step: 1000 | train loss: 0.1862 | f1分数: 87.1132 | time: 148.2355
# Step: 1050 | train loss: 0.2242 | f1分数: 86.6020 | time: 155.6331
# Step: 1100 | train loss: 0.1213 | f1分数: 87.4774 | time: 163.0512
# Step: 1150 | train loss: 0.2382 | f1分数: 87.1959 | time: 170.2792
# Step: 1200 | train loss: 0.1531 | f1分数: 88.3641 | time: 177.5949
# Step: 1250 | train loss: 0.1570 | f1分数: 89.2184 | time: 184.8416
# Step: 1300 | train loss: 0.1010 | f1分数: 89.6214 | time: 192.2180
# Step: 1350 | train loss: 0.2050 | f1分数: 88.9695 | time: 199.6087
# Step: 1400 | train loss: 0.0454 | f1分数: 87.1237 | time: 206.8625
# Step: 1450 | train loss: 0.1140 | f1分数: 87.2485 | time: 214.2302
# Step: 1500 | train loss: 0.0616 | f1分数: 87.6802 | time: 221.6291
# Step: 1550 | train loss: 0.0407 | f1分数: 88.2685 | time: 228.9507
# Step: 1600 | train loss: 0.0607 | f1分数: 89.6889 | time: 236.3522
# Step: 1650 | train loss: 0.1028 | f1分数: 90.6275 | time: 243.7109
# Step: 1700 | train loss: 0.0493 | f1分数: 89.1210 | time: 251.0939
# Step: 1750 | train loss: 0.0815 | f1分数: 89.8913 | time: 258.4679
# Step: 1800 | train loss: 0.0858 | f1分数: 89.5313 | time: 265.8158
# Step: 1850 | train loss: 0.1197 | f1分数: 89.1493 | time: 273.2310
# Step: 1900 | train loss: 0.0572 | f1分数: 88.9855 | time: 280.6288
# Step: 1950 | train loss: 0.0780 | f1分数: 82.3465 | time: 287.9839
# Step: 2000 | train loss: 0.0664 | f1分数: 89.1194 | time: 295.3632
# Step: 2050 | train loss: 0.0687 | f1分数: 83.0735 | time: 302.7211
# Step: 2100 | train loss: 0.0600 | f1分数: 87.3133 | time: 310.1015
# Step: 2150 | train loss: 0.0647 | f1分数: 89.1028 | time: 317.5198
# Step: 2200 | train loss: 0.0423 | f1分数: 84.3943 | time: 324.8840
# Step: 2250 | train loss: 0.0389 | f1分数: 89.4541 | time: 332.2503
# Step: 2300 | train loss: 0.0883 | f1分数: 88.7397 | time: 339.5869
# Step: 2350 | train loss: 0.0186 | f1分数: 90.0362 | time: 346.9423
# Step: 2400 | train loss: 0.0385 | f1分数: 89.1701 | time: 354.2945
# Step: 2450 | train loss: 0.0185 | f1分数: 89.8979 | time: 361.6598
# Step: 2500 | train loss: 0.0751 | f1分数: 90.1022 | time: 369.0434
# Step: 2550 | train loss: 0.0206 | f1分数: 85.1688 | time: 376.4596
# Step: 2600 | train loss: 0.0227 | f1分数: 89.1014 | time: 384.0556
# Step: 2650 | train loss: 0.0660 | f1分数: 89.9495 | time: 391.5786
# Step: 2700 | train loss: 0.0527 | f1分数: 89.6844 | time: 399.2350
# Step: 2750 | train loss: 0.0147 | f1分数: 85.8018 | time: 406.6006
# Step: 2800 | train loss: 0.0124 | f1分数: 89.2811 | time: 413.9007
# Step: 2850 | train loss: 0.0118 | f1分数: 86.7483 | time: 421.1697
# Step: 2900 | train loss: 0.0301 | f1分数: 89.8859 | time: 428.5570
# Step: 2950 | train loss: 0.0683 | f1分数: 89.0070 | time: 435.9391
# Step: 3000 | train loss: 0.0141 | f1分数: 90.2933 | time: 443.3612
# Step: 3050 | train loss: 0.0885 | f1分数: 89.5875 | time: 450.7279
# Step: 3100 | train loss: 0.0366 | f1分数: 89.6551 | time: 458.1388
# Step: 3150 | train loss: 0.0633 | f1分数: 89.6714 | time: 465.5079
# Step: 3200 | train loss: 0.0228 | f1分数: 90.3572 | time: 472.8323
# Step: 3250 | train loss: 0.0101 | f1分数: 90.6036 | time: 480.1528
# Step: 3300 | train loss: 0.0117 | f1分数: 91.1379 | time: 487.4507
# Step: 3350 | train loss: 0.0303 | f1分数: 90.0109 | time: 494.8143
# Step: 3400 | train loss: 0.0193 | f1分数: 90.3786 | time: 502.1348
# Step: 3450 | train loss: 0.0428 | f1分数: 89.0957 | time: 509.5308
# Step: 3500 | train loss: 0.0416 | f1分数: 88.1156 | time: 516.8322
# Step: 3550 | train loss: 0.0597 | f1分数: 86.7788 | time: 524.0726
# Step: 3600 | train loss: 0.0295 | f1分数: 89.5609 | time: 531.3181
# Step: 3650 | train loss: 0.0411 | f1分数: 87.8381 | time: 538.6359
# Step: 3700 | train loss: 0.0067 | f1分数: 89.2959 | time: 546.0468
# Step: 3750 | train loss: 0.0364 | f1分数: 89.5263 | time: 553.3122
# Step: 3800 | train loss: 0.0635 | f1分数: 87.7063 | time: 560.5658
# Step: 3850 | train loss: 0.0225 | f1分数: 86.7206 | time: 567.7860
# Step: 3900 | train loss: 0.0200 | f1分数: 89.6248 | time: 575.0691
# Step: 3950 | train loss: 0.0227 | f1分数: 90.1048 | time: 582.3850
# Step: 4000 | train loss: 0.0592 | f1分数: 89.6546 | time: 589.7428
# Step: 4050 | train loss: 0.0102 | f1分数: 90.0354 | time: 596.9771
# Step: 4100 | train loss: 0.0027 | f1分数: 86.6917 | time: 604.2509
# Step: 4150 | train loss: 0.0235 | f1分数: 90.0933 | time: 611.4862
# Step: 4200 | train loss: 0.0107 | f1分数: 88.8172 | time: 618.6932
# Step: 4250 | train loss: 0.0366 | f1分数: 89.3487 | time: 626.1539
# Step: 4300 | train loss: 0.0424 | f1分数: 88.8646 | time: 633.5234
# Step: 4350 | train loss: 0.0030 | f1分数: 90.1708 | time: 640.9546
# Step: 4400 | train loss: 0.0283 | f1分数: 90.1318 | time: 648.3330
# Step: 4450 | train loss: 0.0035 | f1分数: 89.9372 | time: 655.7369
# Step: 4500 | train loss: 0.0171 | f1分数: 90.3045 | time: 663.1075
# Step: 4550 | train loss: 0.0014 | f1分数: 90.1567 | time: 670.4798
# Step: 4600 | train loss: 0.0041 | f1分数: 90.0347 | time: 677.7434
# Step: 4650 | train loss: 0.0180 | f1分数: 90.5116 | time: 685.0036
# Step: 4700 | train loss: 0.0173 | f1分数: 89.7107 | time: 692.2365
# Step: 4750 | train loss: 0.0067 | f1分数: 89.9603 | time: 699.5354
# Step: 4800 | train loss: 0.0131 | f1分数: 89.3999 | time: 706.8338
# Step: 4850 | train loss: 0.0035 | f1分数: 90.2115 | time: 714.1201
# Step: 4900 | train loss: 0.0025 | f1分数: 89.4466 | time: 721.4234
# Step: 4950 | train loss: 0.0268 | f1分数: 89.8168 | time: 728.7352
# Step: 5000 | train loss: 0.0074 | f1分数: 90.4639 | time: 736.0157
# Step: 5050 | train loss: 0.0039 | f1分数: 90.5609 | time: 743.3174
# Step: 5100 | train loss: 0.0145 | f1分数: 89.9521 | time: 750.5922
# Step: 5150 | train loss: 0.0178 | f1分数: 87.6263 | time: 757.8651
# Step: 5200 | train loss: 0.0165 | f1分数: 87.8198 | time: 765.1495
# Step: 5250 | train loss: 0.0093 | f1分数: 89.8279 | time: 772.4250
# Step: 5300 | train loss: 0.0385 | f1分数: 88.7415 | time: 779.7262
# Step: 5350 | train loss: 0.0336 | f1分数: 90.0844 | time: 787.0045
# Step: 5400 | train loss: 0.0122 | f1分数: 89.8756 | time: 794.3135
# Step: 5450 | train loss: 0.0371 | f1分数: 89.7752 | time: 801.6022
# Step: 5500 | train loss: 0.0248 | f1分数: 89.7264 | time: 808.9065
# Step: 5550 | train loss: 0.0389 | f1分数: 88.4185 | time: 816.1662
# Step: 5600 | train loss: 0.0380 | f1分数: 88.5777 | time: 823.4565
# Step: 5650 | train loss: 0.0125 | f1分数: 89.8037 | time: 830.7120
# Step: 5700 | train loss: 0.0591 | f1分数: 90.3310 | time: 838.0103
# Step: 5750 | train loss: 0.0055 | f1分数: 90.3883 | time: 845.3301
# Step: 5800 | train loss: 0.0201 | f1分数: 91.1709 | time: 852.6118
# Step: 5850 | train loss: 0.0108 | f1分数: 90.8808 | time: 859.9148
# Step: 5900 | train loss: 0.0009 | f1分数: 90.1430 | time: 867.1787
# Step: 5950 | train loss: 0.0017 | f1分数: 90.2789 | time: 874.4794
# Step: 6000 | train loss: 0.0138 | f1分数: 90.1352 | time: 881.7708
# Step: 6050 | train loss: 0.0170 | f1分数: 90.4023 | time: 889.0457
# Step: 6100 | train loss: 0.0042 | f1分数: 90.6623 | time: 896.3429
# Step: 6150 | train loss: 0.0120 | f1分数: 90.0680 | time: 903.6783
# Step: 6200 | train loss: 0.0064 | f1分数: 90.1662 | time: 911.0712
# Step: 6250 | train loss: 0.0016 | f1分数: 90.5550 | time: 918.3792
# Step: 6300 | train loss: 0.0202 | f1分数: 90.3288 | time: 925.6954
# Step: 6350 | train loss: 0.0091 | f1分数: 89.6724 | time: 932.8759
# Step: 6400 | train loss: 0.0013 | f1分数: 90.5060 | time: 940.1345
# Step: 6450 | train loss: 0.0144 | f1分数: 90.5597 | time: 947.3810
# Step: 6500 | train loss: 0.0100 | f1分数: 90.9942 | time: 954.7250
# Step: 6550 | train loss: 0.0053 | f1分数: 90.4215 | time: 961.9576
# Step: 6600 | train loss: 0.0016 | f1分数: 90.9378 | time: 969.2214
# Step: 6650 | train loss: 0.0118 | f1分数: 89.4639 | time: 976.5651
# Step: 6700 | train loss: 0.0136 | f1分数: 90.9403 | time: 983.8609
# Step: 6750 | train loss: 0.0004 | f1分数: 90.5130 | time: 991.1714
# Step: 6800 | train loss: 0.0069 | f1分数: 90.3520 | time: 998.5455
# Step: 6850 | train loss: 0.0004 | f1分数: 90.4636 | time: 1005.9012
# Step: 6900 | train loss: 0.0008 | f1分数: 90.4935 | time: 1013.2551
# Step: 6950 | train loss: 0.0016 | f1分数: 90.0247 | time: 1020.6427
# Step: 7000 | train loss: 0.0024 | f1分数: 90.7745 | time: 1028.0259
# Step: 7050 | train loss: 0.0012 | f1分数: 90.1205 | time: 1035.4329
# Step: 7100 | train loss: 0.0002 | f1分数: 90.8597 | time: 1042.7927
# Step: 7150 | train loss: 0.0100 | f1分数: 89.9965 | time: 1050.1595
# Step: 7200 | train loss: 0.0006 | f1分数: 89.5161 | time: 1057.4168
# Step: 7250 | train loss: 0.0020 | f1分数: 89.2361 | time: 1064.7587
# Step: 7300 | train loss: 0.0452 | f1分数: 89.3912 | time: 1072.1286
# Step: 7350 | train loss: 0.0469 | f1分数: 89.1048 | time: 1079.4522
# Step: 7400 | train loss: 0.0520 | f1分数: 87.7041 | time: 1086.7345
# Step: 7450 | train loss: 0.0606 | f1分数: 87.6490 | time: 1094.1417
# Step: 7500 | train loss: 0.0164 | f1分数: 89.3961 | time: 1101.5084
# Step: 7550 | train loss: 0.0092 | f1分数: 89.5220 | time: 1108.8308
# Step: 7600 | train loss: 0.0390 | f1分数: 90.6499 | time: 1116.1461
# Step: 7650 | train loss: 0.0072 | f1分数: 90.8483 | time: 1123.4949
# Step: 7700 | train loss: 0.0035 | f1分数: 90.9901 | time: 1130.7565
# Step: 7750 | train loss: 0.0079 | f1分数: 90.8439 | time: 1138.0046
# Step: 7800 | train loss: 0.0027 | f1分数: 91.0252 | time: 1145.3020
# Step: 7850 | train loss: 0.0019 | f1分数: 91.1805 | time: 1152.6456
# Step: 7900 | train loss: 0.0002 | f1分数: 91.1099 | time: 1160.0122
# Step: 7950 | train loss: 0.0009 | f1分数: 91.1156 | time: 1167.2866
# Step: 8000 | train loss: 0.0004 | f1分数: 90.9955 | time: 1174.5098
# Step: 8050 | train loss: 0.0031 | f1分数: 91.2258 | time: 1181.7328
# Step: 8100 | train loss: 0.0001 | f1分数: 91.2029 | time: 1188.9601
# Step: 8150 | train loss: 0.0261 | f1分数: 91.1725 | time: 1196.3367
# Step: 8200 | train loss: 0.0197 | f1分数: 91.0794 | time: 1203.5924
# Step: 8250 | train loss: 0.0166 | f1分数: 90.4122 | time: 1210.9359
# Step: 8300 | train loss: 0.0050 | f1分数: 89.4682 | time: 1218.2722
# Step: 8350 | train loss: 0.0316 | f1分数: 90.3756 | time: 1225.6133
# Step: 8400 | train loss: 0.0058 | f1分数: 89.5909 | time: 1232.8867
# Step: 8450 | train loss: 0.0237 | f1分数: 89.9659 | time: 1240.2219
# Step: 8500 | train loss: 0.0094 | f1分数: 90.1637 | time: 1247.4757
# Step: 8550 | train loss: 0.0081 | f1分数: 89.8770 | time: 1254.7394
# Step: 8600 | train loss: 0.0034 | f1分数: 88.6522 | time: 1262.0652
# Step: 8650 | train loss: 0.0219 | f1分数: 90.3331 | time: 1269.3994
# Step: 8700 | train loss: 0.0001 | f1分数: 89.8065 | time: 1276.7308
# Step: 8750 | train loss: 0.0065 | f1分数: 88.8943 | time: 1284.1034
# Step: 8800 | train loss: 0.0047 | f1分数: 89.0973 | time: 1291.5042
# Step: 8850 | train loss: 0.0037 | f1分数: 89.6711 | time: 1298.9910
# Step: 8900 | train loss: 0.0058 | f1分数: 89.1423 | time: 1306.2940
# Step: 8950 | train loss: 0.0288 | f1分数: 90.5417 | time: 1313.7136
# Step: 9000 | train loss: 0.0024 | f1分数: 90.1722 | time: 1321.2194
# Step: 9050 | train loss: 0.0016 | f1分数: 90.5307 | time: 1328.6381
# Step: 9100 | train loss: 0.0362 | f1分数: 90.1788 | time: 1336.0424
# Step: 9150 | train loss: 0.0052 | f1分数: 89.6789 | time: 1343.4984
# Step: 9200 | train loss: 0.0013 | f1分数: 89.3762 | time: 1350.9562
# Step: 9250 | train loss: 0.0015 | f1分数: 90.2097 | time: 1358.4167
# Step: 9300 | train loss: 0.0064 | f1分数: 90.6899 | time: 1365.8530
# Step: 9350 | train loss: 0.0009 | f1分数: 89.9559 | time: 1373.3106
# Step: 9400 | train loss: 0.0004 | f1分数: 90.4237 | time: 1380.7815
# Step: 9450 | train loss: 0.0032 | f1分数: 89.3271 | time: 1388.2610
# Step: 9500 | train loss: 0.0002 | f1分数: 90.1871 | time: 1395.7159
# Step: 9550 | train loss: 0.0006 | f1分数: 90.6328 | time: 1403.1619
# Step: 9600 | train loss: 0.0006 | f1分数: 90.7082 | time: 1410.5858
# Step: 9650 | train loss: 0.0003 | f1分数: 90.7409 | time: 1418.0394
# Step: 9700 | train loss: 0.0002 | f1分数: 90.4745 | time: 1425.4723
# Step: 9750 | train loss: 0.0000 | f1分数: 90.8793 | time: 1432.8849
# Step: 9800 | train loss: 0.0003 | f1分数: 90.8249 | time: 1440.3467
# Step: 9850 | train loss: 0.0002 | f1分数: 90.8248 | time: 1447.7929
# Step: 9900 | train loss: 0.0001 | f1分数: 90.8504 | time: 1455.2672
# Step: 9950 | train loss: 0.0001 | f1分数: 90.8248 | time: 1462.7254
# Step: 10000 | train loss: 0.0001 | f1分数: 90.8979 | time: 1470.2155
# Step: 10050 | train loss: 0.0001 | f1分数: 90.8122 | time: 1477.6523
# Step: 10100 | train loss: 0.0001 | f1分数: 90.8851 | time: 1485.1160
# Step: 10150 | train loss: 0.0001 | f1分数: 90.8702 | time: 1492.5666
# Step: 10200 | train loss: 0.0001 | f1分数: 90.7031 | time: 1499.9904
# Step: 10250 | train loss: 0.0002 | f1分数: 90.6935 | time: 1507.4583
# Step: 10300 | train loss: 0.0001 | f1分数: 90.4048 | time: 1514.9287
# Step: 10350 | train loss: 0.0034 | f1分数: 90.8303 | time: 1522.4077
# Step: 10400 | train loss: 0.0080 | f1分数: 89.2527 | time: 1529.8935
# Step: 10450 | train loss: 0.0066 | f1分数: 89.9174 | time: 1537.3485
# Step: 10500 | train loss: 0.0267 | f1分数: 87.5083 | time: 1544.7646
# Step: 10550 | train loss: 0.0329 | f1分数: 88.9277 | time: 1552.0561
# Step: 10600 | train loss: 0.0423 | f1分数: 83.5524 | time: 1559.3663
# Step: 10650 | train loss: 0.0680 | f1分数: 85.9977 | time: 1566.6419
# Step: 10700 | train loss: 0.0373 | f1分数: 87.7690 | time: 1573.8712
# Step: 10750 | train loss: 0.0973 | f1分数: 87.5823 | time: 1581.1163
# Step: 10800 | train loss: 0.0309 | f1分数: 89.2827 | time: 1588.3849
# Step: 10850 | train loss: 0.0334 | f1分数: 88.3100 | time: 1595.6751
# Step: 10900 | train loss: 0.0166 | f1分数: 89.0866 | time: 1602.9709
# Step: 10950 | train loss: 0.0173 | f1分数: 90.0371 | time: 1610.2342
# Step: 11000 | train loss: 0.0125 | f1分数: 90.8546 | time: 1617.5073
# Step: 11050 | train loss: 0.0008 | f1分数: 90.9695 | time: 1624.7959
# Step: 11100 | train loss: 0.0047 | f1分数: 90.9748 | time: 1631.9644
# Step: 11150 | train loss: 0.0078 | f1分数: 89.7250 | time: 1639.1577
# Step: 11200 | train loss: 0.0083 | f1分数: 90.5709 | time: 1646.3716
# Step: 11250 | train loss: 0.0010 | f1分数: 91.1924 | time: 1653.5258
# Step: 11300 | train loss: 0.0041 | f1分数: 90.6678 | time: 1660.6927
# Step: 11350 | train loss: 0.0010 | f1分数: 90.8248 | time: 1667.8833
# Step: 11400 | train loss: 0.0001 | f1分数: 91.3403 | time: 1675.0667
# Step: 11450 | train loss: 0.0002 | f1分数: 91.3235 | time: 1682.2605
# Step: 11500 | train loss: 0.0010 | f1分数: 91.2156 | time: 1689.4756
# Step: 11550 | train loss: 0.0002 | f1分数: 91.2055 | time: 1696.6516
# Step: 11600 | train loss: 0.0003 | f1分数: 91.2026 | time: 1703.8583
# Step: 11650 | train loss: 0.0001 | f1分数: 91.2631 | time: 1711.0657
# Step: 11700 | train loss: 0.0002 | f1分数: 91.2061 | time: 1718.2910
# Step: 11750 | train loss: 0.0002 | f1分数: 91.2283 | time: 1725.4774
# Step: 11800 | train loss: 0.0003 | f1分数: 91.2963 | time: 1732.6428
# Step: 11850 | train loss: 0.0001 | f1分数: 91.2711 | time: 1739.8244
# Step: 11900 | train loss: 0.0002 | f1分数: 91.1562 | time: 1746.9835
# Step: 11950 | train loss: 0.0001 | f1分数: 91.3366 | time: 1754.2010
# Step: 12000 | train loss: 0.0002 | f1分数: 91.2839 | time: 1761.3933
# Step: 12050 | train loss: 0.0001 | f1分数: 91.1994 | time: 1768.5546
# Step: 12100 | train loss: 0.0008 | f1分数: 91.1922 | time: 1775.7465
# Step: 12150 | train loss: 0.0001 | f1分数: 91.2003 | time: 1782.9146
# Step: 12200 | train loss: 0.0004 | f1分数: 91.1757 | time: 1790.0946
# Step: 12250 | train loss: 0.0002 | f1分数: 91.1343 | time: 1797.2849
# Step: 12300 | train loss: 0.0001 | f1分数: 91.0969 | time: 1804.4593
# Step: 12350 | train loss: 0.0001 | f1分数: 91.1264 | time: 1811.6833
# Step: 12400 | train loss: 0.0000 | f1分数: 91.1617 | time: 1818.8519
# Step: 12450 | train loss: 0.0002 | f1分数: 91.1943 | time: 1826.0339
# Step: 12500 | train loss: 0.0001 | f1分数: 91.2330 | time: 1833.2108
# Step: 12550 | train loss: 0.0000 | f1分数: 91.1117 | time: 1840.3610
# Step: 12600 | train loss: 0.0003 | f1分数: 91.0841 | time: 1847.5374
# Step: 12650 | train loss: 0.0015 | f1分数: 90.0978 | time: 1854.6698
# Step: 12700 | train loss: 0.0034 | f1分数: 90.4314 | time: 1861.8520
# Step: 12750 | train loss: 0.0161 | f1分数: 89.2467 | time: 1869.0278
# Step: 12800 | train loss: 0.0791 | f1分数: 88.2021 | time: 1876.2688
# Step: 12850 | train loss: 0.0344 | f1分数: 88.6330 | time: 1883.4572
# Step: 12900 | train loss: 0.0696 | f1分数: 89.6791 | time: 1890.6158
# Step: 12950 | train loss: 0.0091 | f1分数: 89.9206 | time: 1897.8131
# Step: 13000 | train loss: 0.1056 | f1分数: 90.5926 | time: 1905.0119
# Step: 13050 | train loss: 0.0045 | f1分数: 90.5122 | time: 1912.1782
# Step: 13100 | train loss: 0.0092 | f1分数: 90.9029 | time: 1919.3648
# Step: 13150 | train loss: 0.0067 | f1分数: 91.0276 | time: 1926.5860
# Step: 13200 | train loss: 0.0002 | f1分数: 90.5665 | time: 1933.7809
# Step: 13250 | train loss: 0.0002 | f1分数: 90.8523 | time: 1940.9758
# Step: 13300 | train loss: 0.0004 | f1分数: 90.7441 | time: 1948.1828
# Step: 13350 | train loss: 0.0021 | f1分数: 90.8820 | time: 1955.3611
# Step: 13400 | train loss: 0.0001 | f1分数: 90.8640 | time: 1962.5370
# Step: 13450 | train loss: 0.0069 | f1分数: 90.9005 | time: 1969.7226
# Step: 13500 | train loss: 0.0030 | f1分数: 90.8607 | time: 1976.8909
# Step: 13550 | train loss: 0.0001 | f1分数: 90.9676 | time: 1984.0753
# Step: 13600 | train loss: 0.0002 | f1分数: 90.9773 | time: 1991.2524
# Step: 13650 | train loss: 0.0001 | f1分数: 90.8897 | time: 1998.4300
# Step: 13700 | train loss: 0.0002 | f1分数: 90.8897 | time: 2005.6124
# Step: 13750 | train loss: 0.0003 | f1分数: 90.9933 | time: 2012.7578
# Step: 13800 | train loss: 0.0002 | f1分数: 90.9407 | time: 2019.9359
# Step: 13850 | train loss: 0.0004 | f1分数: 90.8609 | time: 2027.1565
# Step: 13900 | train loss: 0.0001 | f1分数: 90.8829 | time: 2034.3510



