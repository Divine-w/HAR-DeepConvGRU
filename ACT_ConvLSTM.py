import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics

X_train = np.load("./ACT_data/processed/np_train_x.npy")
X_test = np.load("./ACT_data/processed/np_test_x.npy")
y_train = np.load("./ACT_data/processed/np_train_y.npy")
y_test = np.load("./ACT_data/processed/np_test_y.npy")
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
# training_iters = 1000*2000
batch_size = 1000
num_units_lstm = 128
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

def ConvLSTM(xs, is_training):
    conv1 = tf.layers.conv1d(xs, 16, 8, 1, 'same', activation=tf.nn.relu)
    # conv1 = tf.contrib.layers.batch_norm(inputs=conv1,
    #                                      decay=0.90,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    pool1 = tf.layers.max_pooling1d(conv1, 2, 2, padding='same')
    conv2 = tf.layers.conv1d(pool1, 32, 8, 1, 'same', activation=tf.nn.relu)
    # conv2 = tf.contrib.layers.batch_norm(inputs=conv2,
    #                                      decay=0.90,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    pool2 = tf.layers.max_pooling1d(conv2, 2, 2, padding='same')
    conv3 = tf.layers.conv1d(pool2, 64, 8, 1, 'same', activation=tf.nn.relu)
    # conv3 = tf.contrib.layers.batch_norm(inputs=conv3,
    #                                      decay=0.90,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    # conv1 = tf.layers.conv1d(xs, 16, 5, 1, 'same', activation=tf.nn.relu)
    # conv2 = tf.layers.conv1d(conv1, 32, 5, 1, 'same', activation=tf.nn.relu)
    # conv3 = tf.layers.conv1d(conv2, 64, 5, 1, 'same', activation=tf.nn.relu)

    rnn_cell_1 = tf.contrib.rnn.LSTMCell(num_units_lstm)
    rnn_cell_2 = tf.contrib.rnn.LSTMCell(num_units_lstm)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell_1, rnn_cell_2])
    # init_state = rnn_cell.zero_state(_batch_size, dtype=tf.float32)
    outputs, last_states = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        conv3,  # input
        initial_state=None,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
    )

    output = tf.layers.dense(outputs[:, -1, :], n_classes, activation=tf.nn.softmax)  # output based on the last output step

    return output

xs = tf.placeholder(tf.float32, [None, n_steps, n_input],name='input')
ys = tf.placeholder(tf.float32, [None, n_classes],name='label')
is_training = tf.placeholder(tf.bool, name='train')
# _batch_size = tf.placeholder(dtype=tf.int32, shape=[])

output = ConvLSTM(xs, is_training)

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
Time = []
with tf.Session() as sess:
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    step = 1
    start_time = time.time()
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = extract_batch_size(y_train, step, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys, is_training: True})
        train_losses.append(loss_)
        Time.append(time.time() - start_time)
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, is_training: False})
            pred = sess.run(argmax_pred, feed_dict={xs: X_test, ys: y_test, is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_
                  , '| time: %.4f'% (time.time() - start_time))
            print("精度: {:.4f}%".format(100 * metrics.precision_score(argmax_y, pred, average="weighted")))
            print("召回率: {:.4f}%".format(100 * metrics.recall_score(argmax_y, pred, average="weighted")))
        # if step % 1000 == 0 and op == 1:
        #     np.save("./figure/ACT1/lstm_losses_5.npy", train_losses)
        #     np.save("./figure/ACT1/lstm_time_5.npy", Time)
        #     print('损失已保存为numpy文件')
        # if step % 100 == 0 and op == 1 and accuracy_>min_acc:
        #     saver.save(sess, "./ACTmodel1/ConvLSTM_model")
        #     min_acc = accuracy_
        #     print('ConvLSTM模型保存成功')
        # if step % 1000 == 0:
        #     indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
        #     plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        #     plt.show()
        step += 1

# 0初始状态
# Step: 50 | train loss: 0.9510 | test accuracy: 0.7034 | time: 14.2301
# Step: 100 | train loss: 0.8517 | test accuracy: 0.7708 | time: 27.3243
# Step: 150 | train loss: 0.4832 | test accuracy: 0.7743 | time: 40.4867
# Step: 200 | train loss: 0.6632 | test accuracy: 0.8098 | time: 54.1844
# Step: 250 | train loss: 0.6131 | test accuracy: 0.8082 | time: 67.9071
# Step: 300 | train loss: 0.3969 | test accuracy: 0.8357 | time: 81.6248
# Step: 350 | train loss: 0.3885 | test accuracy: 0.8306 | time: 96.4193
# Step: 400 | train loss: 0.6414 | test accuracy: 0.8494 | time: 110.1470
# Step: 450 | train loss: 0.5062 | test accuracy: 0.8501 | time: 123.8957
# Step: 500 | train loss: 0.5793 | test accuracy: 0.8606 | time: 137.5934
# Step: 550 | train loss: 0.2350 | test accuracy: 0.8735 | time: 151.3071
# Step: 600 | train loss: 0.2824 | test accuracy: 0.8761 | time: 164.9568
# Step: 650 | train loss: 0.3311 | test accuracy: 0.8796 | time: 178.6665
# Step: 700 | train loss: 0.7289 | test accuracy: 0.8770 | time: 192.3812
# Step: 750 | train loss: 0.3346 | test accuracy: 0.8961 | time: 206.0549
# Step: 800 | train loss: 0.1701 | test accuracy: 0.8965 | time: 219.7526
# Step: 850 | train loss: 0.1678 | test accuracy: 0.8973 | time: 233.3762
# Step: 900 | train loss: 0.2208 | test accuracy: 0.8974 | time: 247.0819
# Step: 950 | train loss: 0.2121 | test accuracy: 0.9135 | time: 260.7827
# Step: 1000 | train loss: 0.0920 | test accuracy: 0.9188 | time: 274.4693

# Step: 50 | train loss: 0.6635 | test accuracy: 0.7397 | time: 13.3614
# Step: 100 | train loss: 0.7692 | test accuracy: 0.7798 | time: 25.7592
# Step: 150 | train loss: 0.6724 | test accuracy: 0.7417 | time: 38.7174
# Step: 200 | train loss: 0.5648 | test accuracy: 0.8424 | time: 51.4064
# Step: 250 | train loss: 0.5788 | test accuracy: 0.8417 | time: 64.0633
# Step: 300 | train loss: 0.3722 | test accuracy: 0.8499 | time: 77.1606
# Step: 350 | train loss: 0.2694 | test accuracy: 0.8549 | time: 90.2739
# Step: 400 | train loss: 0.5081 | test accuracy: 0.8640 | time: 103.3402
# Step: 450 | train loss: 0.5145 | test accuracy: 0.8473 | time: 116.4644
# Step: 500 | train loss: 0.5841 | test accuracy: 0.8635 | time: 129.5627
# Step: 550 | train loss: 0.2011 | test accuracy: 0.8759 | time: 142.6430
# Step: 600 | train loss: 0.2979 | test accuracy: 0.8872 | time: 155.7012
# Step: 650 | train loss: 0.2750 | test accuracy: 0.8906 | time: 168.8005
# Step: 700 | train loss: 0.5791 | test accuracy: 0.8744 | time: 181.8417
# Step: 750 | train loss: 0.3616 | test accuracy: 0.9037 | time: 194.9750
# Step: 800 | train loss: 0.1806 | test accuracy: 0.8872 | time: 208.0573
# Step: 850 | train loss: 0.1495 | test accuracy: 0.8839 | time: 221.2466
# Step: 900 | train loss: 0.2636 | test accuracy: 0.9217 | time: 234.8383
# Step: 950 | train loss: 0.2095 | test accuracy: 0.9300 | time: 248.0106
# Step: 1000 | train loss: 0.0789 | test accuracy: 0.9456 | time: 261.1529

# 三层128
# Step: 50 | train loss: 0.9785 | test accuracy: 0.6511 | time: 10.0479
# Step: 100 | train loss: 0.7281 | test accuracy: 0.7642 | time: 18.2199
# Step: 150 | train loss: 0.3074 | test accuracy: 0.8364 | time: 25.8987
# Step: 200 | train loss: 0.4882 | test accuracy: 0.8501 | time: 32.5921
# Step: 250 | train loss: 0.5034 | test accuracy: 0.8561 | time: 34.2615
# Step: 300 | train loss: 0.2479 | test accuracy: 0.8401 | time: 35.9646
# Step: 350 | train loss: 0.2349 | test accuracy: 0.8609 | time: 37.6729
# Step: 400 | train loss: 0.3120 | test accuracy: 0.8768 | time: 39.3858
# Step: 450 | train loss: 0.3219 | test accuracy: 0.9021 | time: 41.1142
# Step: 500 | train loss: 0.4444 | test accuracy: 0.9167 | time: 42.8532
# Step: 550 | train loss: 0.0912 | test accuracy: 0.9291 | time: 44.5458
# Step: 600 | train loss: 0.1142 | test accuracy: 0.9166 | time: 46.2035
# Step: 650 | train loss: 0.2866 | test accuracy: 0.9259 | time: 47.8680
# Step: 700 | train loss: 0.3688 | test accuracy: 0.9366 | time: 49.5940
# Step: 750 | train loss: 0.1705 | test accuracy: 0.9499 | time: 51.3028
# Step: 800 | train loss: 0.0889 | test accuracy: 0.9444 | time: 52.9830
# Step: 850 | train loss: 0.0586 | test accuracy: 0.9593 | time: 54.6825
# Step: 900 | train loss: 0.1181 | test accuracy: 0.9291 | time: 56.3465
# Step: 950 | train loss: 0.1602 | test accuracy: 0.9529 | time: 58.0377
# Step: 1000 | train loss: 0.0252 | test accuracy: 0.9593 | time: 59.7368
# Step: 1050 | train loss: 0.0442 | test accuracy: 0.9632 | time: 61.4303
# Step: 1100 | train loss: 0.1072 | test accuracy: 0.9669 | time: 63.1208
# Step: 1150 | train loss: 0.1217 | test accuracy: 0.9591 | time: 64.8321
# Step: 1200 | train loss: 0.1104 | test accuracy: 0.9589 | time: 66.4871
# Step: 1250 | train loss: 0.0521 | test accuracy: 0.9577 | time: 68.1953
# Step: 1300 | train loss: 0.1429 | test accuracy: 0.9456 | time: 69.9121
# Step: 1350 | train loss: 0.0742 | test accuracy: 0.9684 | time: 71.6209
# Step: 1400 | train loss: 0.0501 | test accuracy: 0.9678 | time: 73.3284
# Step: 1450 | train loss: 0.0559 | test accuracy: 0.9658 | time: 75.0415
# Step: 1500 | train loss: 0.0183 | test accuracy: 0.9748 | time: 76.7692
# Step: 1550 | train loss: 0.0756 | test accuracy: 0.9725 | time: 78.3838
# Step: 1600 | train loss: 0.1791 | test accuracy: 0.9528 | time: 80.0573
# Step: 1650 | train loss: 0.0192 | test accuracy: 0.9322 | time: 81.6875
# Step: 1700 | train loss: 0.0606 | test accuracy: 0.9465 | time: 83.3827
# Step: 1750 | train loss: 0.0868 | test accuracy: 0.9612 | time: 85.0353
# Step: 1800 | train loss: 0.0551 | test accuracy: 0.9732 | time: 86.7110
# Step: 1850 | train loss: 0.0618 | test accuracy: 0.9781 | time: 88.4003
# Step: 1900 | train loss: 0.0127 | test accuracy: 0.9803 | time: 90.0816
# Step: 1950 | train loss: 0.0230 | test accuracy: 0.9778 | time: 91.7903
# Step: 2000 | train loss: 0.0171 | test accuracy: 0.9734 | time: 93.4942
# Step: 2050 | train loss: 0.0375 | test accuracy: 0.9785 | time: 95.1683
# Step: 2100 | train loss: 0.0242 | test accuracy: 0.9799 | time: 96.8747
# Step: 2150 | train loss: 0.0066 | test accuracy: 0.9814 | time: 98.5466
# Step: 2200 | train loss: 0.0295 | test accuracy: 0.9821 | time: 100.2127
# Step: 2250 | train loss: 0.0164 | test accuracy: 0.9815 | time: 101.8445
# Step: 2300 | train loss: 0.0037 | test accuracy: 0.9817 | time: 103.5178
# Step: 2350 | train loss: 0.0026 | test accuracy: 0.9810 | time: 105.2184
# Step: 2400 | train loss: 0.0096 | test accuracy: 0.9817 | time: 106.9260
# Step: 2450 | train loss: 0.0046 | test accuracy: 0.9789 | time: 108.6162
# Step: 2500 | train loss: 0.0776 | test accuracy: 0.9696 | time: 110.2825
# Step: 2550 | train loss: 0.0285 | test accuracy: 0.9649 | time: 111.9793
# Step: 2600 | train loss: 0.0364 | test accuracy: 0.9764 | time: 113.6885
# Step: 2650 | train loss: 0.0133 | test accuracy: 0.9803 | time: 115.3706
# Step: 2700 | train loss: 0.0064 | test accuracy: 0.9810 | time: 117.0760
# Step: 2750 | train loss: 0.0075 | test accuracy: 0.9775 | time: 118.7666
# Step: 2800 | train loss: 0.0022 | test accuracy: 0.9822 | time: 120.4688
# Step: 2850 | train loss: 0.0220 | test accuracy: 0.9758 | time: 122.2239
# Step: 2900 | train loss: 0.0122 | test accuracy: 0.9799 | time: 123.8940
# Step: 2950 | train loss: 0.0147 | test accuracy: 0.9499 | time: 125.5898
# Step: 3000 | train loss: 0.0217 | test accuracy: 0.9751 | time: 127.2664
# Step: 3050 | train loss: 0.0071 | test accuracy: 0.9800 | time: 128.9551
# Step: 3100 | train loss: 0.0039 | test accuracy: 0.9834 | time: 130.7206
# Step: 3150 | train loss: 0.0111 | test accuracy: 0.9841 | time: 132.3786
# Step: 3200 | train loss: 0.0009 | test accuracy: 0.9837 | time: 134.0699
# Step: 3250 | train loss: 0.0057 | test accuracy: 0.9827 | time: 135.7207
# Step: 3300 | train loss: 0.0012 | test accuracy: 0.9823 | time: 137.3869
# Step: 3350 | train loss: 0.0009 | test accuracy: 0.9823 | time: 139.0870
# Step: 3400 | train loss: 0.0012 | test accuracy: 0.9825 | time: 140.7812
# Step: 3450 | train loss: 0.0004 | test accuracy: 0.9826 | time: 142.4796
# Step: 3500 | train loss: 0.0015 | test accuracy: 0.9829 | time: 144.1975
# Step: 3550 | train loss: 0.0040 | test accuracy: 0.9815 | time: 145.9065
# Step: 3600 | train loss: 0.0287 | test accuracy: 0.9811 | time: 147.5776
# Step: 3650 | train loss: 0.0028 | test accuracy: 0.9817 | time: 149.2374
# Step: 3700 | train loss: 0.0050 | test accuracy: 0.9837 | time: 150.9081
# Step: 3750 | train loss: 0.0004 | test accuracy: 0.9837 | time: 152.6117
# Step: 3800 | train loss: 0.0011 | test accuracy: 0.9836 | time: 154.2823
# Step: 3850 | train loss: 0.0002 | test accuracy: 0.9827 | time: 156.0150
# Step: 3900 | train loss: 0.0001 | test accuracy: 0.9822 | time: 157.7327
# Step: 3950 | train loss: 0.0004 | test accuracy: 0.9814 | time: 159.4477
# Step: 4000 | train loss: 0.0002 | test accuracy: 0.9822 | time: 161.1600
# Step: 4050 | train loss: 0.0003 | test accuracy: 0.9822 | time: 162.7966
# Step: 4100 | train loss: 0.0002 | test accuracy: 0.9819 | time: 164.4799
# Step: 4150 | train loss: 0.0005 | test accuracy: 0.9830 | time: 166.1780
# Step: 4200 | train loss: 0.0017 | test accuracy: 0.9833 | time: 167.8667
# Step: 4250 | train loss: 0.0257 | test accuracy: 0.9832 | time: 169.5010
# Step: 4300 | train loss: 0.0007 | test accuracy: 0.9826 | time: 171.1763
# Step: 4350 | train loss: 0.0015 | test accuracy: 0.9833 | time: 172.8827
# Step: 4400 | train loss: 0.0001 | test accuracy: 0.9829 | time: 174.5719
# Step: 4450 | train loss: 0.0010 | test accuracy: 0.9826 | time: 176.2155
# Step: 4500 | train loss: 0.0001 | test accuracy: 0.9829 | time: 177.9128
# Step: 4550 | train loss: 0.0000 | test accuracy: 0.9826 | time: 179.5671
# Step: 4600 | train loss: 0.0002 | test accuracy: 0.9819 | time: 181.2206
# Step: 4650 | train loss: 0.0001 | test accuracy: 0.9812 | time: 182.8908
# Step: 4700 | train loss: 0.0004 | test accuracy: 0.9821 | time: 184.5270
# Step: 4750 | train loss: 0.0001 | test accuracy: 0.9826 | time: 186.2213
# Step: 4800 | train loss: 0.0003 | test accuracy: 0.9823 | time: 187.8709
# Step: 4850 | train loss: 0.0001 | test accuracy: 0.9818 | time: 189.5215
# Step: 4900 | train loss: 0.4811 | test accuracy: 0.9745 | time: 191.2368
# Step: 4950 | train loss: 0.5160 | test accuracy: 0.9226 | time: 192.9263
# Step: 5000 | train loss: 0.1426 | test accuracy: 0.9403 | time: 194.5921
# Step: 5050 | train loss: 0.0824 | test accuracy: 0.9737 | time: 196.2308
# Step: 5100 | train loss: 0.1265 | test accuracy: 0.9775 | time: 197.9145

# Step: 50 | train loss: 1.0772 | test accuracy: 0.6254 | time: 2.9742
# Step: 100 | train loss: 0.7741 | test accuracy: 0.7771 | time: 4.8258
# Step: 150 | train loss: 0.4055 | test accuracy: 0.8043 | time: 6.6847
# Step: 200 | train loss: 0.5270 | test accuracy: 0.7715 | time: 8.5108
# Step: 250 | train loss: 0.6020 | test accuracy: 0.8210 | time: 10.2598
# Step: 300 | train loss: 0.3427 | test accuracy: 0.8460 | time: 12.1431
# Step: 350 | train loss: 0.2386 | test accuracy: 0.8655 | time: 13.9775
# Step: 400 | train loss: 0.2710 | test accuracy: 0.8638 | time: 15.8878
# Step: 450 | train loss: 0.5263 | test accuracy: 0.8601 | time: 17.7714
# Step: 500 | train loss: 0.5888 | test accuracy: 0.8853 | time: 19.6389
# Step: 550 | train loss: 0.1774 | test accuracy: 0.8895 | time: 21.4913
# Step: 600 | train loss: 0.2193 | test accuracy: 0.9035 | time: 23.3546
# Step: 650 | train loss: 0.4241 | test accuracy: 0.8885 | time: 25.2132
# Step: 700 | train loss: 0.5510 | test accuracy: 0.9136 | time: 27.0821
# Step: 750 | train loss: 0.3434 | test accuracy: 0.9057 | time: 28.9796
# Step: 800 | train loss: 0.1311 | test accuracy: 0.8966 | time: 30.8335
# Step: 850 | train loss: 0.1176 | test accuracy: 0.9300 | time: 32.7122
# Step: 900 | train loss: 0.1537 | test accuracy: 0.9382 | time: 34.5532
# Step: 950 | train loss: 0.1591 | test accuracy: 0.9408 | time: 36.3860
# Step: 1000 | train loss: 0.0784 | test accuracy: 0.9351 | time: 38.1879
# Step: 1050 | train loss: 0.1206 | test accuracy: 0.9343 | time: 40.0378
# Step: 1100 | train loss: 0.3511 | test accuracy: 0.9359 | time: 41.8832
# Step: 1150 | train loss: 0.2076 | test accuracy: 0.9500 | time: 43.7837
# Step: 1200 | train loss: 0.2394 | test accuracy: 0.9606 | time: 45.6212
# Step: 1250 | train loss: 0.0384 | test accuracy: 0.9630 | time: 47.4155
# Step: 1300 | train loss: 0.1986 | test accuracy: 0.9693 | time: 49.2530
# Step: 1350 | train loss: 0.0318 | test accuracy: 0.9711 | time: 51.1188
# Step: 1400 | train loss: 0.0660 | test accuracy: 0.9730 | time: 52.9760
# Step: 1450 | train loss: 0.0321 | test accuracy: 0.9602 | time: 54.8771
# Step: 1500 | train loss: 0.0304 | test accuracy: 0.9675 | time: 56.7411
# Step: 1550 | train loss: 0.2565 | test accuracy: 0.9552 | time: 58.6029
# Step: 1600 | train loss: 0.1239 | test accuracy: 0.9621 | time: 60.5197
# Step: 1650 | train loss: 0.0237 | test accuracy: 0.9700 | time: 62.3562
# Step: 1700 | train loss: 0.0517 | test accuracy: 0.9685 | time: 64.1495
# Step: 1750 | train loss: 0.0522 | test accuracy: 0.9728 | time: 66.0016
# Step: 1800 | train loss: 0.0419 | test accuracy: 0.9522 | time: 67.8538
# Step: 1850 | train loss: 0.0888 | test accuracy: 0.9747 | time: 69.6976
# Step: 1900 | train loss: 0.0101 | test accuracy: 0.9793 | time: 71.5813
# Step: 1950 | train loss: 0.0383 | test accuracy: 0.9786 | time: 73.4680
# Step: 2000 | train loss: 0.0250 | test accuracy: 0.9788 | time: 75.3213
# Step: 2050 | train loss: 0.0170 | test accuracy: 0.9784 | time: 77.1903
# Step: 2100 | train loss: 0.0106 | test accuracy: 0.9749 | time: 79.0015
# Step: 2150 | train loss: 0.0098 | test accuracy: 0.9771 | time: 80.9277
# Step: 2200 | train loss: 0.1658 | test accuracy: 0.9647 | time: 82.7689
# Step: 2250 | train loss: 0.0376 | test accuracy: 0.9726 | time: 84.6138
# Step: 2300 | train loss: 0.0041 | test accuracy: 0.9804 | time: 86.4818
# Step: 2350 | train loss: 0.0238 | test accuracy: 0.9823 | time: 88.3575
# Step: 2400 | train loss: 0.0116 | test accuracy: 0.9827 | time: 90.2484
# Step: 2450 | train loss: 0.0063 | test accuracy: 0.9803 | time: 92.0693
# Step: 2500 | train loss: 0.0745 | test accuracy: 0.9804 | time: 93.9286
# Step: 2550 | train loss: 0.0899 | test accuracy: 0.9562 | time: 95.8263
# Step: 2600 | train loss: 0.2067 | test accuracy: 0.9762 | time: 97.6819
# Step: 2650 | train loss: 0.0382 | test accuracy: 0.9728 | time: 99.5198
# Step: 2700 | train loss: 0.0453 | test accuracy: 0.9788 | time: 101.3898
# Step: 2750 | train loss: 0.0108 | test accuracy: 0.9665 | time: 103.2753
# Step: 2800 | train loss: 0.0060 | test accuracy: 0.9773 | time: 105.1879
# Step: 2850 | train loss: 0.0521 | test accuracy: 0.9723 | time: 107.0338
# Step: 2900 | train loss: 0.0387 | test accuracy: 0.9748 | time: 108.9140
# Step: 2950 | train loss: 0.0105 | test accuracy: 0.9721 | time: 110.7419
# Step: 3000 | train loss: 0.0092 | test accuracy: 0.9833 | time: 112.5982
# Step: 3050 | train loss: 0.0103 | test accuracy: 0.9838 | time: 114.4696
# Step: 3100 | train loss: 0.0020 | test accuracy: 0.9840 | time: 116.3275
# Step: 3150 | train loss: 0.0296 | test accuracy: 0.9840 | time: 118.1588
# Step: 3200 | train loss: 0.0010 | test accuracy: 0.9841 | time: 119.9496
# Step: 3250 | train loss: 0.0041 | test accuracy: 0.9844 | time: 121.8233
# Step: 3300 | train loss: 0.0010 | test accuracy: 0.9843 | time: 123.6917
# Step: 3350 | train loss: 0.0011 | test accuracy: 0.9841 | time: 125.5335
# Step: 3400 | train loss: 0.0025 | test accuracy: 0.9830 | time: 127.3953
# Step: 3450 | train loss: 0.0006 | test accuracy: 0.9847 | time: 129.2154
# Step: 3500 | train loss: 0.0015 | test accuracy: 0.9844 | time: 131.0267
# Step: 3550 | train loss: 0.0011 | test accuracy: 0.9843 | time: 132.8668
# Step: 3600 | train loss: 0.0254 | test accuracy: 0.9837 | time: 134.7473
# Step: 3650 | train loss: 0.0006 | test accuracy: 0.9843 | time: 136.5515
# Step: 3700 | train loss: 0.0023 | test accuracy: 0.9841 | time: 138.4035
# Step: 3750 | train loss: 0.0005 | test accuracy: 0.9840 | time: 140.2441
# Step: 3800 | train loss: 0.0190 | test accuracy: 0.9840 | time: 142.1269
# Step: 3850 | train loss: 0.0003 | test accuracy: 0.9840 | time: 144.0434
# Step: 3900 | train loss: 0.0002 | test accuracy: 0.9841 | time: 145.9580
# Step: 3950 | train loss: 0.0003 | test accuracy: 0.9837 | time: 147.8072
# Step: 4000 | train loss: 0.0002 | test accuracy: 0.9837 | time: 149.6980
# Step: 4050 | train loss: 0.0003 | test accuracy: 0.9840 | time: 151.5442
# Step: 4100 | train loss: 0.0069 | test accuracy: 0.9829 | time: 153.3715
# Step: 4150 | train loss: 0.1130 | test accuracy: 0.9580 | time: 155.2766
# Step: 4200 | train loss: 0.0614 | test accuracy: 0.9738 | time: 157.1080
# Step: 4250 | train loss: 0.0456 | test accuracy: 0.9785 | time: 158.9290
# Step: 4300 | train loss: 0.0633 | test accuracy: 0.9606 | time: 160.7787
# Step: 4350 | train loss: 0.0136 | test accuracy: 0.9822 | time: 162.6309
# Step: 4400 | train loss: 0.0040 | test accuracy: 0.9851 | time: 164.4178
# Step: 4450 | train loss: 0.0169 | test accuracy: 0.9851 | time: 166.2525
# Step: 4500 | train loss: 0.0023 | test accuracy: 0.9843 | time: 168.1440
# Step: 4550 | train loss: 0.0008 | test accuracy: 0.9834 | time: 170.0149
# Step: 4600 | train loss: 0.0009 | test accuracy: 0.9843 | time: 171.8093
# Step: 4650 | train loss: 0.0004 | test accuracy: 0.9843 | time: 173.6755
# Step: 4700 | train loss: 0.0008 | test accuracy: 0.9844 | time: 175.5142
# Step: 4750 | train loss: 0.0003 | test accuracy: 0.9845 | time: 177.4334
# Step: 4800 | train loss: 0.0005 | test accuracy: 0.9849 | time: 179.2986
# Step: 4850 | train loss: 0.0012 | test accuracy: 0.9847 | time: 181.1737
# Step: 4900 | train loss: 0.0287 | test accuracy: 0.9847 | time: 183.0287
# Step: 4950 | train loss: 0.0020 | test accuracy: 0.9845 | time: 184.8689
# Step: 5000 | train loss: 0.0046 | test accuracy: 0.9771 | time: 186.7338
# Step: 5050 | train loss: 0.1995 | test accuracy: 0.9421 | time: 188.6104
# Step: 5100 | train loss: 0.2172 | test accuracy: 0.9682 | time: 190.4349