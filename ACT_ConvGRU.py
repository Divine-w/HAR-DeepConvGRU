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
# training_iters = 1000 * 2000
batch_size = 1000
num_units_gru = 128
n_classes = 6
min_acc = 0


def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def ConvGRU(xs, is_training):
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

    rnn_cell_1 = tf.contrib.rnn.GRUCell(num_units_gru)
    rnn_cell_2 = tf.contrib.rnn.GRUCell(num_units_gru)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell_1, rnn_cell_2])
    # init_state = rnn_cell.zero_state(_batch_size, dtype=tf.float32)
    outputs, last_states = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        conv3,  # input
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
# _batch_size = tf.placeholder(dtype=tf.int32, shape=[])

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
with tf.Session() as sess:
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
            accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, is_training: False})
            pred = sess.run(argmax_pred, feed_dict={xs: X_test, ys: y_test, is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_
                  , '| time: %.4f' % (time.time() - start_time))
            print("精度: {:.4f}%".format(100 * metrics.precision_score(argmax_y, pred, average="weighted")))
            print("召回率: {:.4f}%".format(100 * metrics.recall_score(argmax_y, pred, average="weighted")))
        # if step % 1000 == 0 and op == 1:
        #     np.save("./figure/ACT1/gru_losses_5.npy", train_losses)
        #     np.save("./figure/ACT1/gru_time_5.npy", Time)
        #     print('损失已保存为numpy文件')
        # if step % 100 == 0 and op == 1 and accuracy_ > min_acc:
        #     saver.save(sess, "./ACTmodel/ConvGRU_model")
        #     min_acc = accuracy_
        #     print('ConvGRU模型保存成功')
        # if step % 1000 == 0:
        #     indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
        #     plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        #     plt.show()
        step += 1

        # 两层cnn
        # Step: 50 | train loss: 0.8411 | test accuracy: 0.6823 | time: 17.4413
        # Step: 100 | train loss: 0.8248 | test accuracy: 0.7809 | time: 30.6094
        # Step: 150 | train loss: 0.4761 | test accuracy: 0.8209 | time: 43.6250
        # Step: 200 | train loss: 0.5115 | test accuracy: 0.8347 | time: 56.8247
        # Step: 250 | train loss: 0.5073 | test accuracy: 0.8477 | time: 70.5656
        # Step: 300 | train loss: 0.2977 | test accuracy: 0.8557 | time: 83.8418
        # Step: 350 | train loss: 0.2313 | test accuracy: 0.8599 | time: 97.3437
        # Step: 400 | train loss: 0.5347 | test accuracy: 0.8602 | time: 110.5889
        # Step: 450 | train loss: 0.3881 | test accuracy: 0.8650 | time: 123.8708
        # Step: 500 | train loss: 0.6123 | test accuracy: 0.8846 | time: 137.1311
        # Step: 550 | train loss: 0.1940 | test accuracy: 0.8940 | time: 150.6200
        # Step: 600 | train loss: 0.1756 | test accuracy: 0.9047 | time: 163.9078
        # Step: 650 | train loss: 0.3989 | test accuracy: 0.9110 | time: 177.5029
        # Step: 700 | train loss: 0.5316 | test accuracy: 0.9095 | time: 190.6952
        # Step: 750 | train loss: 0.2341 | test accuracy: 0.9254 | time: 203.9302
        # Step: 800 | train loss: 0.0749 | test accuracy: 0.9272 | time: 217.2039
        # Step: 850 | train loss: 0.0789 | test accuracy: 0.9399 | time: 230.3691
        # Step: 900 | train loss: 0.1353 | test accuracy: 0.9422 | time: 243.6124
        # Step: 950 | train loss: 0.1543 | test accuracy: 0.9454 | time: 256.8244
        # Step: 1000 | train loss: 0.0582 | test accuracy: 0.9484 | time: 270.1083
        # Step: 1050 | train loss: 0.1147 | test accuracy: 0.9371 | time: 286.4910
        # Step: 1100 | train loss: 0.2698 | test accuracy: 0.9450 | time: 299.7866
        # Step: 1150 | train loss: 0.3024 | test accuracy: 0.9563 | time: 313.1546
        # Step: 1200 | train loss: 0.1830 | test accuracy: 0.9547 | time: 326.5158
        # Step: 1250 | train loss: 0.0608 | test accuracy: 0.9621 | time: 340.4284
        # Step: 1300 | train loss: 0.2096 | test accuracy: 0.9556 | time: 353.7417
        # Step: 1350 | train loss: 0.1006 | test accuracy: 0.9448 | time: 367.1564
        # Step: 1400 | train loss: 0.1817 | test accuracy: 0.9507 | time: 380.5667
        # Step: 1450 | train loss: 0.0678 | test accuracy: 0.9469 | time: 393.9349
        # Step: 1500 | train loss: 0.0687 | test accuracy: 0.9552 | time: 407.3084
        # Step: 1550 | train loss: 0.1915 | test accuracy: 0.9452 | time: 420.6313
        # Step: 1600 | train loss: 0.1503 | test accuracy: 0.9681 | time: 433.9182
        # Step: 1650 | train loss: 0.0270 | test accuracy: 0.9699 | time: 447.5879
        # Step: 1700 | train loss: 0.0723 | test accuracy: 0.9708 | time: 461.9358
        # Step: 1750 | train loss: 0.0718 | test accuracy: 0.9714 | time: 475.5562
        # Step: 1800 | train loss: 0.1305 | test accuracy: 0.9715 | time: 488.8840
        # Step: 1850 | train loss: 0.1267 | test accuracy: 0.9721 | time: 502.1610
        # Step: 1900 | train loss: 0.0325 | test accuracy: 0.9699 | time: 515.6281
        # Step: 1950 | train loss: 0.0816 | test accuracy: 0.9681 | time: 533.2284
        # Step: 2000 | train loss: 0.0317 | test accuracy: 0.9681 | time: 546.6021
        # Step: 2050 | train loss: 0.1190 | test accuracy: 0.9515 | time: 563.0941
        # Step: 2100 | train loss: 0.0464 | test accuracy: 0.9673 | time: 577.8737
        # Step: 2150 | train loss: 0.0292 | test accuracy: 0.9655 | time: 591.4154
        # Step: 2200 | train loss: 0.1076 | test accuracy: 0.9536 | time: 604.7549
        # Step: 2250 | train loss: 0.0675 | test accuracy: 0.9688 | time: 618.0806
        # Step: 2300 | train loss: 0.0203 | test accuracy: 0.9665 | time: 631.4371
        # Step: 2350 | train loss: 0.0518 | test accuracy: 0.9701 | time: 644.8121
        # Step: 2400 | train loss: 0.0435 | test accuracy: 0.9712 | time: 658.6292
        # Step: 2450 | train loss: 0.0345 | test accuracy: 0.9756 | time: 671.9915
        # Step: 2500 | train loss: 0.1360 | test accuracy: 0.9774 | time: 685.3252
        # Step: 2550 | train loss: 0.0249 | test accuracy: 0.9780 | time: 698.7203
        # Step: 2600 | train loss: 0.0523 | test accuracy: 0.9782 | time: 712.0819
        # Step: 2650 | train loss: 0.0184 | test accuracy: 0.9780 | time: 725.4842
        # Step: 2700 | train loss: 0.0252 | test accuracy: 0.9773 | time: 738.7975
        # Step: 2750 | train loss: 0.0135 | test accuracy: 0.9770 | time: 752.1699
        # Step: 2800 | train loss: 0.0050 | test accuracy: 0.9778 | time: 765.5227
        # Step: 2850 | train loss: 0.0423 | test accuracy: 0.9764 | time: 778.8884
        # Step: 2900 | train loss: 0.1070 | test accuracy: 0.9762 | time: 792.2800
        # Step: 2950 | train loss: 0.2897 | test accuracy: 0.9002 | time: 805.5959
        # Step: 3000 | train loss: 0.0686 | test accuracy: 0.9737 | time: 818.9779

        # 0初始状态
        # Step: 50 | train loss: 0.7233 | test accuracy: 0.6933 | time: 13.6457
        # Step: 100 | train loss: 0.8214 | test accuracy: 0.7993 | time: 26.2196
        # Step: 150 | train loss: 0.4362 | test accuracy: 0.8182 | time: 38.8895
        # Step: 200 | train loss: 0.4821 | test accuracy: 0.8458 | time: 51.6416
        # Step: 250 | train loss: 0.3925 | test accuracy: 0.8480 | time: 64.3726
        # Step: 300 | train loss: 0.3548 | test accuracy: 0.8454 | time: 77.3868
        # Step: 350 | train loss: 0.2303 | test accuracy: 0.8720 | time: 90.6042
        # Step: 400 | train loss: 0.4423 | test accuracy: 0.8702 | time: 103.8495
        # Step: 450 | train loss: 0.3658 | test accuracy: 0.8770 | time: 117.0389
        # Step: 500 | train loss: 0.5101 | test accuracy: 0.8888 | time: 130.3673
        # Step: 550 | train loss: 0.1720 | test accuracy: 0.8951 | time: 143.6195
        # Step: 600 | train loss: 0.2076 | test accuracy: 0.9061 | time: 156.8799
        # Step: 650 | train loss: 0.3972 | test accuracy: 0.9265 | time: 170.1573
        # Step: 700 | train loss: 0.8961 | test accuracy: 0.8673 | time: 183.3726
        # Step: 750 | train loss: 0.2550 | test accuracy: 0.9307 | time: 197.0503
        # Step: 800 | train loss: 0.1133 | test accuracy: 0.9361 | time: 210.3377
        # Step: 850 | train loss: 0.1182 | test accuracy: 0.8952 | time: 223.6001
        # Step: 900 | train loss: 0.1914 | test accuracy: 0.9485 | time: 237.0586
        # Step: 950 | train loss: 0.1753 | test accuracy: 0.9502 | time: 250.3030
        # Step: 1000 | train loss: 0.0632 | test accuracy: 0.9547 | time: 264.0257

        # 三层cnn units = 32
        # Step: 50 | train loss: 0.8283 | test accuracy: 0.7231 | time: 20.2935
        # Step: 100 | train loss: 0.6935 | test accuracy: 0.7357 | time: 38.4652
        # Step: 150 | train loss: 0.3748 | test accuracy: 0.8227 | time: 56.7677
        # Step: 200 | train loss: 0.4322 | test accuracy: 0.8429 | time: 75.4833
        # Step: 250 | train loss: 0.5721 | test accuracy: 0.8553 | time: 93.6958
        # Step: 300 | train loss: 0.3014 | test accuracy: 0.8714 | time: 112.0957
        # Step: 350 | train loss: 0.1829 | test accuracy: 0.8847 | time: 130.3349
        # Step: 400 | train loss: 0.3550 | test accuracy: 0.8903 | time: 148.8103
        # Step: 450 | train loss: 0.2847 | test accuracy: 0.9089 | time: 167.1146
        # Step: 500 | train loss: 0.4931 | test accuracy: 0.9173 | time: 185.2902
        # Step: 550 | train loss: 0.1293 | test accuracy: 0.9221 | time: 203.6178
        # Step: 600 | train loss: 0.1251 | test accuracy: 0.9267 | time: 221.9343
        # Step: 650 | train loss: 0.3397 | test accuracy: 0.9413 | time: 240.2866
        # Step: 700 | train loss: 0.4959 | test accuracy: 0.9437 | time: 258.6792
        # Step: 750 | train loss: 0.1669 | test accuracy: 0.9511 | time: 277.6545
        # Step: 800 | train loss: 0.0712 | test accuracy: 0.9502 | time: 296.0254
        # Step: 850 | train loss: 0.0534 | test accuracy: 0.9532 | time: 314.4968
        # Step: 900 | train loss: 0.1160 | test accuracy: 0.9537 | time: 332.7883
        # Step: 950 | train loss: 0.1197 | test accuracy: 0.9566 | time: 351.1448
        # Step: 1000 | train loss: 0.0434 | test accuracy: 0.9558 | time: 369.5515
        # Step: 1050 | train loss: 0.0666 | test accuracy: 0.9628 | time: 393.9950
        # Step: 1100 | train loss: 0.1430 | test accuracy: 0.9652 | time: 412.3012
        # Step: 1150 | train loss: 0.2651 | test accuracy: 0.9603 | time: 430.5843
        # Step: 1200 | train loss: 0.1939 | test accuracy: 0.9652 | time: 448.9087
        # Step: 1250 | train loss: 0.0471 | test accuracy: 0.9685 | time: 467.2476
        # Step: 1300 | train loss: 0.0925 | test accuracy: 0.9630 | time: 485.4595
        # Step: 1350 | train loss: 0.0326 | test accuracy: 0.9610 | time: 505.5139
        # Step: 1400 | train loss: 0.0684 | test accuracy: 0.9489 | time: 523.8253
        # Step: 1450 | train loss: 0.0831 | test accuracy: 0.9358 | time: 542.1980
        # Step: 1500 | train loss: 0.0822 | test accuracy: 0.9534 | time: 560.6228
        # Step: 1550 | train loss: 0.6746 | test accuracy: 0.9221 | time: 578.8532
        # Step: 1600 | train loss: 0.1403 | test accuracy: 0.9634 | time: 597.1457
        # Step: 1650 | train loss: 0.0255 | test accuracy: 0.9728 | time: 615.4117
        # Step: 1700 | train loss: 0.0661 | test accuracy: 0.9714 | time: 633.8954
        # Step: 1750 | train loss: 0.0711 | test accuracy: 0.9766 | time: 652.3537
        # Step: 1800 | train loss: 0.0370 | test accuracy: 0.9752 | time: 670.7462
        # Step: 1850 | train loss: 0.0991 | test accuracy: 0.9749 | time: 689.1810
        # Step: 1900 | train loss: 0.0196 | test accuracy: 0.9793 | time: 713.1206
        # Step: 1950 | train loss: 0.0322 | test accuracy: 0.9801 | time: 734.0303
        # Step: 2000 | train loss: 0.0244 | test accuracy: 0.9807 | time: 756.9627
        # Step: 2050 | train loss: 0.0260 | test accuracy: 0.9796 | time: 782.4496
        # Step: 2100 | train loss: 0.0114 | test accuracy: 0.9792 | time: 803.4703
        # Step: 2150 | train loss: 0.0059 | test accuracy: 0.9774 | time: 828.0069
        # Step: 2200 | train loss: 0.0252 | test accuracy: 0.9758 | time: 850.2829
        # Step: 2250 | train loss: 0.0229 | test accuracy: 0.9715 | time: 875.6131
        # Step: 2300 | train loss: 0.0075 | test accuracy: 0.9723 | time: 897.2977
        # Step: 2350 | train loss: 0.0355 | test accuracy: 0.9686 | time: 916.4163
        # Step: 2400 | train loss: 0.1520 | test accuracy: 0.9485 | time: 935.0237
        # Step: 2450 | train loss: 0.0559 | test accuracy: 0.9685 | time: 953.6627
        # Step: 2500 | train loss: 0.0766 | test accuracy: 0.9812 | time: 971.9873
        # Step: 2550 | train loss: 0.0146 | test accuracy: 0.9818 | time: 990.3097
        # Step: 2600 | train loss: 0.0251 | test accuracy: 0.9781 | time: 1009.1025
        # Step: 2650 | train loss: 0.0059 | test accuracy: 0.9793 | time: 1027.6136
        # Step: 2700 | train loss: 0.0139 | test accuracy: 0.9819 | time: 1045.8211
        # Step: 2750 | train loss: 0.0110 | test accuracy: 0.9747 | time: 1064.1985
        # Step: 2800 | train loss: 0.0041 | test accuracy: 0.9781 | time: 1082.5532
        # Step: 2850 | train loss: 0.1337 | test accuracy: 0.9722 | time: 1100.7018
        # Step: 2900 | train loss: 0.0959 | test accuracy: 0.9537 | time: 1118.9804
        # Step: 2950 | train loss: 0.0181 | test accuracy: 0.9766 | time: 1137.2541
        # Step: 3000 | train loss: 0.0426 | test accuracy: 0.9796 | time: 1155.4771

        # units = 128
        # Step: 50 | train loss: 0.6971 | test accuracy: 0.7431 | time: 44.0337
        # Step: 100 | train loss: 0.6825 | test accuracy: 0.7779 | time: 85.5057
        # Step: 150 | train loss: 0.3101 | test accuracy: 0.7674 | time: 126.7487
        # Step: 200 | train loss: 0.4031 | test accuracy: 0.8603 | time: 167.6175
        # Step: 250 | train loss: 0.3288 | test accuracy: 0.8768 | time: 209.3165
        # Step: 300 | train loss: 0.2214 | test accuracy: 0.8095 | time: 250.4221
        # Step: 350 | train loss: 0.2629 | test accuracy: 0.8462 | time: 291.6770
        # Step: 400 | train loss: 0.3712 | test accuracy: 0.8931 | time: 332.7336
        # Step: 450 | train loss: 0.2576 | test accuracy: 0.9036 | time: 374.6153
        # Step: 500 | train loss: 0.4569 | test accuracy: 0.8806 | time: 418.3587
        # Step: 550 | train loss: 0.1956 | test accuracy: 0.9063 | time: 461.4506
        # Step: 600 | train loss: 0.0879 | test accuracy: 0.9439 | time: 503.1563
        # Step: 650 | train loss: 0.2226 | test accuracy: 0.9469 | time: 546.0919
        # Step: 700 | train loss: 0.2008 | test accuracy: 0.9603 | time: 601.8879
        # Step: 750 | train loss: 0.1067 | test accuracy: 0.9569 | time: 644.2991
        # Step: 800 | train loss: 0.1192 | test accuracy: 0.9425 | time: 689.0132
        # Step: 850 | train loss: 0.0532 | test accuracy: 0.9495 | time: 734.2397
        # Step: 900 | train loss: 0.1210 | test accuracy: 0.9606 | time: 781.3697
        # Step: 950 | train loss: 0.1009 | test accuracy: 0.9632 | time: 831.5019
        # Step: 1000 | train loss: 0.0283 | test accuracy: 0.9648 | time: 874.8369
        # Step: 1050 | train loss: 0.0707 | test accuracy: 0.9606 | time: 926.0148
        # Step: 1100 | train loss: 0.0879 | test accuracy: 0.9673 | time: 968.9585
        # Step: 1150 | train loss: 0.1594 | test accuracy: 0.9722 | time: 1010.4049
        # Step: 1200 | train loss: 0.0999 | test accuracy: 0.9748 | time: 1051.6298
        # Step: 1250 | train loss: 0.0306 | test accuracy: 0.9697 | time: 1094.9173
        # Step: 1300 | train loss: 0.1376 | test accuracy: 0.9567 | time: 1136.7872
        # Step: 1350 | train loss: 0.0315 | test accuracy: 0.9612 | time: 1177.7541
        # Step: 1400 | train loss: 0.0480 | test accuracy: 0.9540 | time: 1220.8374
        # Step: 1450 | train loss: 0.0423 | test accuracy: 0.9649 | time: 1264.4376
        # Step: 1500 | train loss: 0.0894 | test accuracy: 0.9504 | time: 1305.9935
        # Step: 1550 | train loss: 0.2038 | test accuracy: 0.9593 | time: 1347.0064
        # Step: 1600 | train loss: 0.1203 | test accuracy: 0.9682 | time: 1388.8520
        # Step: 1650 | train loss: 0.0299 | test accuracy: 0.9715 | time: 1433.0385
        # Step: 1700 | train loss: 0.0511 | test accuracy: 0.9766 | time: 1474.4505
        # Step: 1750 | train loss: 0.0250 | test accuracy: 0.9814 | time: 1515.7275
        # Step: 1800 | train loss: 0.0166 | test accuracy: 0.9810 | time: 1557.0023
        # Step: 1850 | train loss: 0.0363 | test accuracy: 0.9637 | time: 1599.2851
        # Step: 1900 | train loss: 0.0041 | test accuracy: 0.9812 | time: 1642.4313
        # Step: 1950 | train loss: 0.0337 | test accuracy: 0.9766 | time: 1687.2277
        # Step: 2000 | train loss: 0.0246 | test accuracy: 0.9732 | time: 1728.4423
        # Step: 2050 | train loss: 0.0161 | test accuracy: 0.9751 | time: 1774.8872
        # Step: 2100 | train loss: 0.0187 | test accuracy: 0.9751 | time: 1816.0993
        # Step: 2150 | train loss: 0.0036 | test accuracy: 0.9804 | time: 1857.3706
        # Step: 2200 | train loss: 0.0399 | test accuracy: 0.9752 | time: 1898.5683
        # Step: 2250 | train loss: 0.0123 | test accuracy: 0.9796 | time: 1940.2721
        # Step: 2300 | train loss: 0.0025 | test accuracy: 0.9789 | time: 1981.5148
        # Step: 2350 | train loss: 0.0042 | test accuracy: 0.9811 | time: 2026.2663
        # Step: 2400 | train loss: 0.0017 | test accuracy: 0.9818 | time: 2069.4547
        # Step: 2450 | train loss: 0.0010 | test accuracy: 0.9821 | time: 2111.0355
        # Step: 2500 | train loss: 0.0034 | test accuracy: 0.9818 | time: 2153.0892
        # Step: 2550 | train loss: 0.0005 | test accuracy: 0.9811 | time: 2195.8409
        # Step: 2600 | train loss: 0.0075 | test accuracy: 0.9807 | time: 2237.1674
        # Step: 2650 | train loss: 0.0004 | test accuracy: 0.9808 | time: 2279.0116
        # Step: 2700 | train loss: 0.0005 | test accuracy: 0.9806 | time: 2320.5916
        # Step: 2750 | train loss: 0.0010 | test accuracy: 0.9804 | time: 2363.2294
        # Step: 2800 | train loss: 0.0002 | test accuracy: 0.9807 | time: 2418.0113
        # Step: 2850 | train loss: 0.0012 | test accuracy: 0.9806 | time: 2472.7182
        # Step: 2900 | train loss: 0.0023 | test accuracy: 0.9811 | time: 2523.5209
        # Step: 2950 | train loss: 0.0008 | test accuracy: 0.9804 | time: 2572.9185
        # Step: 3000 | train loss: 0.0031 | test accuracy: 0.9763 | time: 2623.6630

        # Step: 50 | train loss: 0.6575 | test accuracy: 0.7832 | time: 43.4316
        # Step: 100 | train loss: 0.7899 | test accuracy: 0.7947 | time: 84.0478
        # Step: 150 | train loss: 0.2315 | test accuracy: 0.8516 | time: 124.7693
        # Step: 200 | train loss: 0.3485 | test accuracy: 0.8662 | time: 165.4294
        # Step: 250 | train loss: 0.4084 | test accuracy: 0.8748 | time: 206.1723
        # Step: 300 | train loss: 0.2985 | test accuracy: 0.8673 | time: 246.7992
        # Step: 350 | train loss: 0.1319 | test accuracy: 0.8781 | time: 287.4325
        # Step: 400 | train loss: 0.6170 | test accuracy: 0.8709 | time: 328.1563
        # Step: 450 | train loss: 0.3360 | test accuracy: 0.8688 | time: 368.9783
        # Step: 500 | train loss: 0.4674 | test accuracy: 0.9225 | time: 410.3564
        # Step: 550 | train loss: 0.1432 | test accuracy: 0.9244 | time: 451.3037
        # Step: 600 | train loss: 0.1718 | test accuracy: 0.9051 | time: 492.2639
        # Step: 650 | train loss: 0.2463 | test accuracy: 0.9411 | time: 533.6392
        # Step: 700 | train loss: 0.3330 | test accuracy: 0.9495 | time: 576.1044
        # Step: 750 | train loss: 0.2106 | test accuracy: 0.9544 | time: 619.8481
        # Step: 800 | train loss: 0.0445 | test accuracy: 0.9618 | time: 660.4776
        # Step: 850 | train loss: 0.0328 | test accuracy: 0.9647 | time: 701.2512
        # Step: 900 | train loss: 0.1013 | test accuracy: 0.9675 | time: 741.9466
        # Step: 950 | train loss: 0.0884 | test accuracy: 0.9669 | time: 782.7213
        # Step: 1000 | train loss: 0.0407 | test accuracy: 0.9413 | time: 823.6830
        # Step: 1050 | train loss: 0.0563 | test accuracy: 0.9539 | time: 864.5544
        # Step: 1100 | train loss: 0.1333 | test accuracy: 0.9540 | time: 905.3030
        # Step: 1150 | train loss: 0.1242 | test accuracy: 0.9685 | time: 946.1247
        # Step: 1200 | train loss: 0.1452 | test accuracy: 0.9563 | time: 987.0505
        # Step: 1250 | train loss: 0.0353 | test accuracy: 0.9675 | time: 1028.0992
        # Step: 1300 | train loss: 0.1329 | test accuracy: 0.9677 | time: 1069.3366
        # Step: 1350 | train loss: 0.0689 | test accuracy: 0.9770 | time: 1110.3603
        # Step: 1400 | train loss: 0.0313 | test accuracy: 0.9775 | time: 1152.2395
        # Step: 1450 | train loss: 0.0212 | test accuracy: 0.9781 | time: 1192.9201
        # Step: 1500 | train loss: 0.0119 | test accuracy: 0.9654 | time: 1233.5068
        # Step: 1550 | train loss: 0.1040 | test accuracy: 0.9714 | time: 1274.2031
        # Step: 1600 | train loss: 0.0813 | test accuracy: 0.9654 | time: 1316.9857
        # Step: 1650 | train loss: 0.0141 | test accuracy: 0.9751 | time: 1358.3880
        # Step: 1700 | train loss: 0.0485 | test accuracy: 0.9681 | time: 1397.9478
        # Step: 1750 | train loss: 0.0459 | test accuracy: 0.9734 | time: 1437.4761
        # Step: 1800 | train loss: 0.0254 | test accuracy: 0.9769 | time: 1477.1986
        # Step: 1850 | train loss: 0.0467 | test accuracy: 0.9578 | time: 1517.7972
        # Step: 1900 | train loss: 0.0099 | test accuracy: 0.9656 | time: 1557.6662
        # Step: 1950 | train loss: 0.0272 | test accuracy: 0.9754 | time: 1597.4291
        # Step: 2000 | train loss: 0.1967 | test accuracy: 0.9576 | time: 1637.2939
        # Step: 2050 | train loss: 0.1102 | test accuracy: 0.9674 | time: 1677.2070
        # Step: 2100 | train loss: 0.0686 | test accuracy: 0.9700 | time: 1716.9743
        # Step: 2150 | train loss: 0.0108 | test accuracy: 0.9810 | time: 1757.7576
        # Step: 2200 | train loss: 0.0236 | test accuracy: 0.9821 | time: 1797.5384
        # Step: 2250 | train loss: 0.0131 | test accuracy: 0.9829 | time: 1838.4985
        # Step: 2300 | train loss: 0.0040 | test accuracy: 0.9829 | time: 1878.4094
        # Step: 2350 | train loss: 0.0081 | test accuracy: 0.9836 | time: 1918.1994
        # Step: 2400 | train loss: 0.0031 | test accuracy: 0.9836 | time: 1959.6046
        # Step: 2450 | train loss: 0.0019 | test accuracy: 0.9833 | time: 2002.7079
        # Step: 2500 | train loss: 0.0067 | test accuracy: 0.9822 | time: 2042.6117
        # Step: 2550 | train loss: 0.0074 | test accuracy: 0.9788 | time: 2082.5694
        # Step: 2600 | train loss: 0.1020 | test accuracy: 0.9829 | time: 2122.4428
        # Step: 2650 | train loss: 0.0336 | test accuracy: 0.9726 | time: 2162.2976
        # Step: 2700 | train loss: 0.0563 | test accuracy: 0.9788 | time: 2202.3661
        # Step: 2750 | train loss: 0.0114 | test accuracy: 0.9759 | time: 2242.3793
        # Step: 2800 | train loss: 0.0010 | test accuracy: 0.9841 | time: 2282.6901
        # Step: 2850 | train loss: 0.0027 | test accuracy: 0.9843 | time: 2327.5689
        # Step: 2900 | train loss: 0.0023 | test accuracy: 0.9840 | time: 2372.5311
        # Step: 2950 | train loss: 0.0012 | test accuracy: 0.9841 | time: 2417.5765
        # Step: 3000 | train loss: 0.0022 | test accuracy: 0.9837 | time: 2459.1763
        # Step: 3050 | train loss: 0.0025 | test accuracy: 0.9834 | time: 2500.0462
        # Step: 3100 | train loss: 0.0005 | test accuracy: 0.9830 | time: 2541.7731
        # Step: 3150 | train loss: 0.0011 | test accuracy: 0.9834 | time: 2583.9973
        # Step: 3200 | train loss: 0.0001 | test accuracy: 0.9819 | time: 2626.9977
        # Step: 3250 | train loss: 0.0006 | test accuracy: 0.9830 | time: 2669.9870
        # Step: 3300 | train loss: 0.0028 | test accuracy: 0.9819 | time: 2712.2309
        # Step: 3350 | train loss: 0.0940 | test accuracy: 0.9545 | time: 2755.7320
        # Step: 3400 | train loss: 0.0265 | test accuracy: 0.9775 | time: 2795.6611
        # Step: 3450 | train loss: 0.0025 | test accuracy: 0.9827 | time: 2837.8267
        # Step: 3500 | train loss: 0.1230 | test accuracy: 0.9667 | time: 2877.8295
        # Step: 3550 | train loss: 0.0384 | test accuracy: 0.9706 | time: 2917.7516
        # Step: 3600 | train loss: 0.0372 | test accuracy: 0.9784 | time: 2958.1320
        # Step: 3650 | train loss: 0.0097 | test accuracy: 0.9819 | time: 2997.9433
        # Step: 3700 | train loss: 0.0052 | test accuracy: 0.9848 | time: 3037.9678
        # Step: 3750 | train loss: 0.0016 | test accuracy: 0.9852 | time: 3077.7974
        # Step: 3800 | train loss: 0.0027 | test accuracy: 0.9852 | time: 3117.4718
        # Step: 3850 | train loss: 0.0005 | test accuracy: 0.9849 | time: 3157.2077
        # Step: 3900 | train loss: 0.0002 | test accuracy: 0.9845 | time: 3197.0837
        # Step: 3950 | train loss: 0.0004 | test accuracy: 0.9847 | time: 3237.7263
        # Step: 4000 | train loss: 0.0004 | test accuracy: 0.9845 | time: 3277.8325
        # Step: 4050 | train loss: 0.0003 | test accuracy: 0.9840 | time: 3318.4420
        # Step: 4100 | train loss: 0.0003 | test accuracy: 0.9845 | time: 3358.0922
        # Step: 4150 | train loss: 0.0004 | test accuracy: 0.9844 | time: 3397.9988
        # Step: 4200 | train loss: 0.0004 | test accuracy: 0.9841 | time: 3438.1266
        # Step: 4250 | train loss: 0.0231 | test accuracy: 0.9848 | time: 3478.5173
        # Step: 4300 | train loss: 0.0026 | test accuracy: 0.9841 | time: 3518.5064
        # Step: 4350 | train loss: 0.0002 | test accuracy: 0.9845 | time: 3561.1835
        # Step: 4400 | train loss: 0.0001 | test accuracy: 0.9845 | time: 3616.8951
        # Step: 4450 | train loss: 0.0005 | test accuracy: 0.9840 | time: 3659.2338
        # Step: 4500 | train loss: 0.0006 | test accuracy: 0.9841 | time: 3703.0322
        # Step: 4550 | train loss: 0.0000 | test accuracy: 0.9841 | time: 3747.3744
        # Step: 4600 | train loss: 0.0001 | test accuracy: 0.9849 | time: 3791.9437
        # Step: 4650 | train loss: 0.0001 | test accuracy: 0.9838 | time: 3836.0154
        # Step: 4700 | train loss: 0.0002 | test accuracy: 0.9847 | time: 3880.1353
        # Step: 4750 | train loss: 0.0001 | test accuracy: 0.9829 | time: 3924.5998
        # Step: 4800 | train loss: 0.0004 | test accuracy: 0.9843 | time: 3965.8022
        # Step: 4850 | train loss: 0.0001 | test accuracy: 0.9838 | time: 4005.8595
        # Step: 4900 | train loss: 0.0200 | test accuracy: 0.9847 | time: 4045.8306
        # Step: 4950 | train loss: 0.0009 | test accuracy: 0.9814 | time: 4090.8368
        # Step: 5000 | train loss: 0.0809 | test accuracy: 0.9540 | time: 4133.2946
        # Step: 5050 | train loss: 0.0956 | test accuracy: 0.9640 | time: 4176.2424
        # Step: 5100 | train loss: 0.2140 | test accuracy: 0.9775 | time: 4219.3918
