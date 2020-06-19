import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

X_train = np.load("./ACT_data/processed/np_train_x.npy")
X_test = np.load("./ACT_data/processed/np_test_x.npy")
y_train = np.load("./ACT_data/processed/np_train_y.npy")
y_test = np.load("./ACT_data/processed/np_test_y.npy")
# print('是否保存模型以便在测试时进行调用：1 是 2 否')
# op = int(input('请选择：'))
start_time = time.time()

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])

lr = 0.001
training_iters = training_data_count * 300
batch_size = 1000
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

def conv1d(xs, is_training):
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
    shape = conv3.get_shape().as_list()
    flat = tf.reshape(conv3, [-1, shape[1] * shape[2]])
    fc1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1, units=128, activation=tf.nn.relu)
    output = tf.layers.dense(fc2, n_classes, activation=tf.nn.softmax)  # output based on the last output step

    return output

xs = tf.placeholder(tf.float32, [None, n_steps, n_input],name='input')
ys = tf.placeholder(tf.float32, [None, n_classes],name='label')
is_training = tf.placeholder(tf.bool)

output = conv1d(xs, is_training)

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
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = extract_batch_size(y_train, step, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys, is_training: True})
        train_losses.append(loss_)
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_
                  , '| time: %.4f' % (time.time() - start_time))
        # if step % 100 == 0 and op == 1 and accuracy_>min_acc:
        #     saver.save(sess, "./ConvLSTMmodel/ConvLSTM_model")
        #     min_acc = accuracy_
        #     print('ConvLSTM模型保存成功')
        # if step % 1000 == 0:
        #     indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
        #     plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        #     plt.show()
        step += 1

# 两层cnn
# Step: 50 | train loss: 0.8521 | test accuracy: 0.6881 | time: 7.3431
# Step: 100 | train loss: 0.8190 | test accuracy: 0.7424 | time: 14.0812
# Step: 150 | train loss: 0.4642 | test accuracy: 0.8193 | time: 20.8341
# Step: 200 | train loss: 0.5374 | test accuracy: 0.8444 | time: 27.5955
# Step: 250 | train loss: 0.4099 | test accuracy: 0.8617 | time: 34.3386
# Step: 300 | train loss: 0.3239 | test accuracy: 0.8683 | time: 41.0544
# Step: 350 | train loss: 0.1758 | test accuracy: 0.8750 | time: 47.7904
# Step: 400 | train loss: 0.2995 | test accuracy: 0.8872 | time: 54.5328
# Step: 450 | train loss: 0.3033 | test accuracy: 0.8947 | time: 61.2730
# Step: 500 | train loss: 0.5017 | test accuracy: 0.9052 | time: 68.0405
# Step: 550 | train loss: 0.1366 | test accuracy: 0.9191 | time: 74.7758
# Step: 600 | train loss: 0.1131 | test accuracy: 0.9265 | time: 81.5647
# Step: 650 | train loss: 0.4271 | test accuracy: 0.9354 | time: 89.1687
# Step: 700 | train loss: 0.4035 | test accuracy: 0.9392 | time: 96.2586
# Step: 750 | train loss: 0.2461 | test accuracy: 0.9393 | time: 103.4703
# Step: 800 | train loss: 0.0917 | test accuracy: 0.9410 | time: 110.6228
# Step: 850 | train loss: 0.0704 | test accuracy: 0.9441 | time: 117.7091
# Step: 900 | train loss: 0.1320 | test accuracy: 0.9459 | time: 124.8018
# Step: 950 | train loss: 0.1588 | test accuracy: 0.9450 | time: 131.9245
# Step: 1000 | train loss: 0.0579 | test accuracy: 0.9418 | time: 139.0248
# Step: 1050 | train loss: 0.1277 | test accuracy: 0.9411 | time: 150.5242
# Step: 1100 | train loss: 0.1940 | test accuracy: 0.9465 | time: 157.4627
# Step: 1150 | train loss: 0.2953 | test accuracy: 0.9484 | time: 164.2010
# Step: 1200 | train loss: 0.2071 | test accuracy: 0.9496 | time: 170.9600
# Step: 1250 | train loss: 0.0620 | test accuracy: 0.9529 | time: 177.7496
# Step: 1300 | train loss: 0.2355 | test accuracy: 0.9510 | time: 184.4823
# Step: 1350 | train loss: 0.1178 | test accuracy: 0.9500 | time: 191.2365
# Step: 1400 | train loss: 0.1622 | test accuracy: 0.9569 | time: 197.9624
# Step: 1450 | train loss: 0.0935 | test accuracy: 0.9552 | time: 204.6939
# Step: 1500 | train loss: 0.0726 | test accuracy: 0.9487 | time: 211.4133
# Step: 1550 | train loss: 0.1601 | test accuracy: 0.9463 | time: 218.1162
# Step: 1600 | train loss: 0.1593 | test accuracy: 0.9473 | time: 224.8304
# Step: 1650 | train loss: 0.0465 | test accuracy: 0.9519 | time: 231.5841
# Step: 1700 | train loss: 0.0880 | test accuracy: 0.9550 | time: 238.3088
# Step: 1750 | train loss: 0.1139 | test accuracy: 0.9541 | time: 245.0184
# Step: 1800 | train loss: 0.0743 | test accuracy: 0.9528 | time: 252.0024
# Step: 1850 | train loss: 0.1877 | test accuracy: 0.9547 | time: 259.1035
# Step: 1900 | train loss: 0.0428 | test accuracy: 0.9610 | time: 266.2136
# Step: 1950 | train loss: 0.1238 | test accuracy: 0.9632 | time: 273.3344
# Step: 2000 | train loss: 0.0835 | test accuracy: 0.9632 | time: 280.4337
# Step: 2050 | train loss: 0.1061 | test accuracy: 0.9640 | time: 292.4218
# Step: 2100 | train loss: 0.0598 | test accuracy: 0.9660 | time: 299.2189
# Step: 2150 | train loss: 0.0326 | test accuracy: 0.9648 | time: 305.8795
# Step: 2200 | train loss: 0.0894 | test accuracy: 0.9640 | time: 312.6153
# Step: 2250 | train loss: 0.0660 | test accuracy: 0.9641 | time: 319.3212
# Step: 2300 | train loss: 0.0247 | test accuracy: 0.9648 | time: 326.3660
# Step: 2350 | train loss: 0.0590 | test accuracy: 0.9651 | time: 333.4759
# Step: 2400 | train loss: 0.0406 | test accuracy: 0.9651 | time: 340.6163
# Step: 2450 | train loss: 0.0235 | test accuracy: 0.9655 | time: 347.7962
# Step: 2500 | train loss: 0.1761 | test accuracy: 0.9666 | time: 354.9008
# Step: 2550 | train loss: 0.0426 | test accuracy: 0.9670 | time: 362.0586
# Step: 2600 | train loss: 0.0912 | test accuracy: 0.9677 | time: 369.1766
# Step: 2650 | train loss: 0.0407 | test accuracy: 0.9670 | time: 376.2298
# Step: 2700 | train loss: 0.0589 | test accuracy: 0.9660 | time: 383.3031
# Step: 2750 | train loss: 0.0276 | test accuracy: 0.9686 | time: 390.3798
# Step: 2800 | train loss: 0.0136 | test accuracy: 0.9691 | time: 397.4448
# Step: 2850 | train loss: 0.0671 | test accuracy: 0.9675 | time: 404.5151
# Step: 2900 | train loss: 0.0506 | test accuracy: 0.9639 | time: 411.6082
# Step: 2950 | train loss: 0.0149 | test accuracy: 0.9628 | time: 418.6908
# Step: 3000 | train loss: 0.0496 | test accuracy: 0.9641 | time: 425.7615

# 三层cnn
# Step: 50 | train loss: 0.8770 | test accuracy: 0.7131 | time: 14.8581
# Step: 100 | train loss: 0.7138 | test accuracy: 0.7049 | time: 25.9830
# Step: 150 | train loss: 0.4128 | test accuracy: 0.8088 | time: 36.9305
# Step: 200 | train loss: 0.5047 | test accuracy: 0.8271 | time: 47.9284
# Step: 250 | train loss: 0.4227 | test accuracy: 0.8532 | time: 59.0644
# Step: 300 | train loss: 0.2260 | test accuracy: 0.8898 | time: 70.7393
# Step: 350 | train loss: 0.1500 | test accuracy: 0.9028 | time: 82.3241
# Step: 400 | train loss: 0.2388 | test accuracy: 0.9041 | time: 93.9935
# Step: 450 | train loss: 0.2469 | test accuracy: 0.9181 | time: 105.5835
# Step: 500 | train loss: 0.4719 | test accuracy: 0.9109 | time: 117.2518
# Step: 550 | train loss: 0.0898 | test accuracy: 0.9147 | time: 128.8928
# Step: 600 | train loss: 0.1567 | test accuracy: 0.9226 | time: 140.7161
# Step: 650 | train loss: 0.1932 | test accuracy: 0.9514 | time: 152.5873
# Step: 700 | train loss: 0.1822 | test accuracy: 0.9462 | time: 164.1686
# Step: 750 | train loss: 0.1690 | test accuracy: 0.9437 | time: 175.8039
# Step: 800 | train loss: 0.0240 | test accuracy: 0.9608 | time: 187.4191
# Step: 850 | train loss: 0.0315 | test accuracy: 0.9584 | time: 199.0359
# Step: 900 | train loss: 0.0662 | test accuracy: 0.9510 | time: 210.7641
# Step: 950 | train loss: 0.0893 | test accuracy: 0.9541 | time: 222.3538
# Step: 1000 | train loss: 0.0146 | test accuracy: 0.9537 | time: 234.0630
# Step: 1050 | train loss: 0.0404 | test accuracy: 0.9545 | time: 616.5406
# Step: 1100 | train loss: 0.0967 | test accuracy: 0.9555 | time: 628.6688
# Step: 1150 | train loss: 0.0253 | test accuracy: 0.9428 | time: 640.3495
# Step: 1200 | train loss: 0.1043 | test accuracy: 0.9519 | time: 652.1165
# Step: 1250 | train loss: 0.0138 | test accuracy: 0.9670 | time: 663.7502
# Step: 1300 | train loss: 0.1276 | test accuracy: 0.9343 | time: 675.4006
# Step: 1350 | train loss: 0.1275 | test accuracy: 0.9463 | time: 686.9758
# Step: 1400 | train loss: 0.0191 | test accuracy: 0.9667 | time: 698.6414
# Step: 1450 | train loss: 0.0052 | test accuracy: 0.9695 | time: 710.2911
# Step: 1500 | train loss: 0.0023 | test accuracy: 0.9681 | time: 721.9567
# Step: 1550 | train loss: 0.0110 | test accuracy: 0.9693 | time: 733.5735
# Step: 1600 | train loss: 0.0214 | test accuracy: 0.9697 | time: 745.2172
# Step: 1650 | train loss: 0.0008 | test accuracy: 0.9701 | time: 756.9092
# Step: 1700 | train loss: 0.0037 | test accuracy: 0.9704 | time: 768.4499
# Step: 1750 | train loss: 0.0015 | test accuracy: 0.9707 | time: 780.0237
# Step: 1800 | train loss: 0.0017 | test accuracy: 0.9706 | time: 791.6044
# Step: 1850 | train loss: 0.0015 | test accuracy: 0.9708 | time: 803.2643
# Step: 1900 | train loss: 0.0004 | test accuracy: 0.9700 | time: 814.8544
# Step: 1950 | train loss: 0.0014 | test accuracy: 0.9715 | time: 826.5274
# Step: 2000 | train loss: 0.0005 | test accuracy: 0.9712 | time: 838.2190
# Step: 2050 | train loss: 0.0004 | test accuracy: 0.9711 | time: 1127.7301
# Step: 2100 | train loss: 0.0005 | test accuracy: 0.9707 | time: 1139.4171
# Step: 2150 | train loss: 0.0001 | test accuracy: 0.9711 | time: 1151.0047
# Step: 2200 | train loss: 0.0004 | test accuracy: 0.9712 | time: 1163.2067
# Step: 2250 | train loss: 0.0021 | test accuracy: 0.9707 | time: 1174.9957
# Step: 2300 | train loss: 0.0001 | test accuracy: 0.9712 | time: 1187.0136
# Step: 2350 | train loss: 0.0019 | test accuracy: 0.9708 | time: 1198.6025
# Step: 2400 | train loss: 0.0002 | test accuracy: 0.9707 | time: 1210.2143
# Step: 2450 | train loss: 0.0005 | test accuracy: 0.9707 | time: 1221.7476
# Step: 2500 | train loss: 0.0004 | test accuracy: 0.9704 | time: 1233.3139
# Step: 2550 | train loss: 0.0001 | test accuracy: 0.9711 | time: 1245.0497
# Step: 2600 | train loss: 0.0003 | test accuracy: 0.9706 | time: 1256.6425
# Step: 2650 | train loss: 0.0002 | test accuracy: 0.9708 | time: 1268.3827
# Step: 2700 | train loss: 0.0001 | test accuracy: 0.9706 | time: 1279.9781
# Step: 2750 | train loss: 0.0003 | test accuracy: 0.9712 | time: 1291.5827
# Step: 2800 | train loss: 0.0000 | test accuracy: 0.9708 | time: 1303.2663
# Step: 2850 | train loss: 0.0001 | test accuracy: 0.9708 | time: 1315.5667
# Step: 2900 | train loss: 0.0011 | test accuracy: 0.9704 | time: 1327.1679
# Step: 2950 | train loss: 0.0000 | test accuracy: 0.9703 | time: 1338.8110
# Step: 3000 | train loss: 0.0018 | test accuracy: 0.9708 | time: 1350.3787

# Step: 50 | train loss: 0.7674 | test accuracy: 0.6852 | time: 2.1785
# Step: 100 | train loss: 0.6118 | test accuracy: 0.8117 | time: 2.6076
# Step: 150 | train loss: 0.2804 | test accuracy: 0.8305 | time: 3.0516
# Step: 200 | train loss: 0.3810 | test accuracy: 0.8546 | time: 3.5050
# Step: 250 | train loss: 0.2398 | test accuracy: 0.8779 | time: 3.9557
# Step: 300 | train loss: 0.2096 | test accuracy: 0.9124 | time: 4.3957
# Step: 350 | train loss: 0.1205 | test accuracy: 0.9065 | time: 4.8455
# Step: 400 | train loss: 0.2212 | test accuracy: 0.8799 | time: 5.2949
# Step: 450 | train loss: 0.1987 | test accuracy: 0.9455 | time: 5.7565
# Step: 500 | train loss: 0.3679 | test accuracy: 0.9289 | time: 6.2027
# Step: 550 | train loss: 0.1383 | test accuracy: 0.8933 | time: 6.6565
# Step: 600 | train loss: 0.0448 | test accuracy: 0.9481 | time: 7.0970
# Step: 650 | train loss: 0.2252 | test accuracy: 0.9602 | time: 7.5444
# Step: 700 | train loss: 0.1142 | test accuracy: 0.9670 | time: 7.9845
# Step: 750 | train loss: 0.0698 | test accuracy: 0.9654 | time: 8.4295
# Step: 800 | train loss: 0.0347 | test accuracy: 0.9669 | time: 8.8782
# Step: 850 | train loss: 0.0202 | test accuracy: 0.9658 | time: 9.3261
# Step: 900 | train loss: 0.0358 | test accuracy: 0.9691 | time: 9.7753
# Step: 950 | train loss: 0.0606 | test accuracy: 0.9593 | time: 10.2209
# Step: 1000 | train loss: 0.0277 | test accuracy: 0.9632 | time: 10.6741
# Step: 1050 | train loss: 0.0476 | test accuracy: 0.9478 | time: 11.1334
# Step: 1100 | train loss: 0.1128 | test accuracy: 0.9484 | time: 11.5800
# Step: 1150 | train loss: 0.0869 | test accuracy: 0.9706 | time: 12.0292
# Step: 1200 | train loss: 0.0770 | test accuracy: 0.9659 | time: 12.4749
# Step: 1250 | train loss: 0.0110 | test accuracy: 0.9728 | time: 12.9298
# Step: 1300 | train loss: 0.0400 | test accuracy: 0.9678 | time: 13.3781
# Step: 1350 | train loss: 0.0041 | test accuracy: 0.9737 | time: 13.8197
# Step: 1400 | train loss: 0.0245 | test accuracy: 0.9688 | time: 14.2654
# Step: 1450 | train loss: 0.0122 | test accuracy: 0.9584 | time: 14.7125
# Step: 1500 | train loss: 0.0053 | test accuracy: 0.9722 | time: 15.1582
# Step: 1550 | train loss: 0.0057 | test accuracy: 0.9751 | time: 15.6096
# Step: 1600 | train loss: 0.0229 | test accuracy: 0.9752 | time: 16.0484
# Step: 1650 | train loss: 0.0012 | test accuracy: 0.9756 | time: 16.5041
# Step: 1700 | train loss: 0.0047 | test accuracy: 0.9754 | time: 16.9565
# Step: 1750 | train loss: 0.0013 | test accuracy: 0.9756 | time: 17.4157
# Step: 1800 | train loss: 0.0010 | test accuracy: 0.9754 | time: 17.8675
# Step: 1850 | train loss: 0.0012 | test accuracy: 0.9751 | time: 18.3100
# Step: 1900 | train loss: 0.0004 | test accuracy: 0.9752 | time: 18.7580
# Step: 1950 | train loss: 0.0038 | test accuracy: 0.9751 | time: 19.2195
# Step: 2000 | train loss: 0.0003 | test accuracy: 0.9744 | time: 19.6639
# Step: 2050 | train loss: 0.0003 | test accuracy: 0.9744 | time: 20.1104
# Step: 2100 | train loss: 0.0012 | test accuracy: 0.9744 | time: 20.5604
# Step: 2150 | train loss: 0.0001 | test accuracy: 0.9743 | time: 21.0191
# Step: 2200 | train loss: 0.0006 | test accuracy: 0.9743 | time: 21.4722
# Step: 2250 | train loss: 0.0004 | test accuracy: 0.9738 | time: 21.9272
# Step: 2300 | train loss: 0.0003 | test accuracy: 0.9741 | time: 22.3729
# Step: 2350 | train loss: 0.0011 | test accuracy: 0.9740 | time: 22.8236
# Step: 2400 | train loss: 0.0001 | test accuracy: 0.9740 | time: 23.2726
# Step: 2450 | train loss: 0.0001 | test accuracy: 0.9741 | time: 23.7168
# Step: 2500 | train loss: 0.0003 | test accuracy: 0.9741 | time: 24.1724
# Step: 2550 | train loss: 0.0001 | test accuracy: 0.9738 | time: 24.6183
# Step: 2600 | train loss: 0.0008 | test accuracy: 0.9704 | time: 25.0626
# Step: 2650 | train loss: 0.2377 | test accuracy: 0.9094 | time: 25.5228
# Step: 2700 | train loss: 0.0963 | test accuracy: 0.9487 | time: 25.9857
# Step: 2750 | train loss: 0.0488 | test accuracy: 0.9597 | time: 26.4298
# Step: 2800 | train loss: 0.0041 | test accuracy: 0.9671 | time: 26.8931
# Step: 2850 | train loss: 0.0222 | test accuracy: 0.9703 | time: 27.3424
# Step: 2900 | train loss: 0.0054 | test accuracy: 0.9752 | time: 27.8106
# Step: 2950 | train loss: 0.0004 | test accuracy: 0.9749 | time: 28.2566
# Step: 3000 | train loss: 0.0018 | test accuracy: 0.9749 | time: 28.7094
# Step: 3050 | train loss: 0.0014 | test accuracy: 0.9747 | time: 29.1650
# Step: 3100 | train loss: 0.0005 | test accuracy: 0.9758 | time: 29.6141
# Step: 3150 | train loss: 0.0014 | test accuracy: 0.9754 | time: 30.0553
# Step: 3200 | train loss: 0.0002 | test accuracy: 0.9758 | time: 30.5135
# Step: 3250 | train loss: 0.0006 | test accuracy: 0.9759 | time: 30.9658
# Step: 3300 | train loss: 0.0001 | test accuracy: 0.9758 | time: 31.4133
# Step: 3350 | train loss: 0.0001 | test accuracy: 0.9760 | time: 31.8656
# Step: 3400 | train loss: 0.0002 | test accuracy: 0.9756 | time: 32.3134
# Step: 3450 | train loss: 0.0001 | test accuracy: 0.9759 | time: 32.7634
# Step: 3500 | train loss: 0.0004 | test accuracy: 0.9759 | time: 33.2241
# Step: 3550 | train loss: 0.0001 | test accuracy: 0.9760 | time: 33.6781
# Step: 3600 | train loss: 0.0003 | test accuracy: 0.9756 | time: 34.1182
# Step: 3650 | train loss: 0.0002 | test accuracy: 0.9755 | time: 34.5545
# Step: 3700 | train loss: 0.0002 | test accuracy: 0.9758 | time: 35.0035
# Step: 3750 | train loss: 0.0001 | test accuracy: 0.9756 | time: 35.4537
# Step: 3800 | train loss: 0.0002 | test accuracy: 0.9756 | time: 35.9062
# Step: 3850 | train loss: 0.0000 | test accuracy: 0.9756 | time: 36.3566
# Step: 3900 | train loss: 0.0000 | test accuracy: 0.9756 | time: 36.8095
# Step: 3950 | train loss: 0.0000 | test accuracy: 0.9756 | time: 37.2613
# Step: 4000 | train loss: 0.0000 | test accuracy: 0.9756 | time: 37.7037
# Step: 4050 | train loss: 0.0000 | test accuracy: 0.9755 | time: 38.1552
# Step: 4100 | train loss: 0.0000 | test accuracy: 0.9756 | time: 38.6039
# Step: 4150 | train loss: 0.0001 | test accuracy: 0.9755 | time: 39.0577
# Step: 4200 | train loss: 0.0000 | test accuracy: 0.9755 | time: 39.5071
# Step: 4250 | train loss: 0.0002 | test accuracy: 0.9758 | time: 39.9508
# Step: 4300 | train loss: 0.0001 | test accuracy: 0.9756 | time: 40.3817
# Step: 4350 | train loss: 0.0001 | test accuracy: 0.9758 | time: 40.8216
# Step: 4400 | train loss: 0.0000 | test accuracy: 0.9759 | time: 41.2837
# Step: 4450 | train loss: 0.0001 | test accuracy: 0.9759 | time: 41.7366
# Step: 4500 | train loss: 0.0000 | test accuracy: 0.9759 | time: 42.1824
# Step: 4550 | train loss: 0.0000 | test accuracy: 0.9759 | time: 42.6251
# Step: 4600 | train loss: 0.0000 | test accuracy: 0.9758 | time: 43.0756
# Step: 4650 | train loss: 0.0000 | test accuracy: 0.9759 | time: 43.5242
# Step: 4700 | train loss: 0.0000 | test accuracy: 0.9759 | time: 43.9679
# Step: 4750 | train loss: 0.0000 | test accuracy: 0.9758 | time: 44.4344
# Step: 4800 | train loss: 0.0000 | test accuracy: 0.9758 | time: 44.8882
# Step: 4850 | train loss: 0.0000 | test accuracy: 0.9758 | time: 45.3299
# Step: 4900 | train loss: 0.0001 | test accuracy: 0.9758 | time: 45.7862
# Step: 4950 | train loss: 0.0000 | test accuracy: 0.9758 | time: 46.2303
# Step: 5000 | train loss: 0.0000 | test accuracy: 0.9758 | time: 46.6765
# Step: 5050 | train loss: 0.0000 | test accuracy: 0.9758 | time: 47.1271
# Step: 5100 | train loss: 0.0000 | test accuracy: 0.9758 | time: 47.5731