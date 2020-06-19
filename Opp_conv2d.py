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

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])

lr = 0.001
training_iters = training_data_count * 300
batch_size = 1000

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
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def CNN(xs, is_training):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
    # conv1 = tf.layers.conv2d(xs, 16, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2], padding='same')
    # conv2 = tf.layers.conv2d(pool1, 32, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2], padding='same')
    # conv3 = tf.layers.conv2d(pool2, 64, [2, 5], 1, 'valid', activation=tf.nn.relu)
    # pool3 = tf.layers.max_pooling2d(conv3, [2, 4], [2, 4], padding='same')
    # shape = pool3.get_shape().as_list()
    # print('dense input shape: {}'.format(shape))
    # flat = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    conv1 = tf.layers.conv2d(xs, 16, [1, 8], 1, 'same', activation=tf.nn.relu)
    # conv1 = tf.contrib.layers.batch_norm(inputs=conv1,
    #                                      decay=0.95,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    pool1 = tf.layers.max_pooling2d(conv1, [2, 1], [2, 1], padding='same')
    conv2 = tf.layers.conv2d(pool1, 32, [1, 8], 1, 'same', activation=tf.nn.relu)
    # conv2 = tf.contrib.layers.batch_norm(inputs=conv2,
    #                                      decay=0.95,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    pool2 = tf.layers.max_pooling2d(conv2, [2, 1], [2, 1], padding='same')
    conv3 = tf.layers.conv2d(pool2, 64, [1, 8], 1, 'same', activation=tf.nn.relu)
    # conv3 = tf.contrib.layers.batch_norm(inputs=conv3,
    #                                      decay=0.95,
    #                                      center=True,
    #                                      scale=True,
    #                                      is_training=is_training,
    #                                      updates_collections=None)
    # pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2], padding='same')
    shape = conv3.get_shape().as_list()
    print('dense input shape: {}'.format(shape))
    flat = tf.reshape(conv3, [-1, shape[1] * shape[2] * shape[3]])
    fc1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1, units=128, activation=tf.nn.relu)
    output = tf.layers.dense(fc2, n_classes,
                             activation=tf.nn.softmax)  # output based on the last output step

    return output

xs = tf.placeholder(tf.float32, [None, n_steps, n_input],name='input')
ys = tf.placeholder(tf.float32, [None, n_classes],name='label')
is_training = tf.placeholder(tf.bool)

output = CNN(xs, is_training)

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
    init_op = tf.group(tf.global_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    step = 1
    start_time = time.time()
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = extract_batch_size(y_train, step, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys,  is_training: True})
        train_losses.append(loss_)
        if step % 50 == 0:
            test_pred = np.empty((0))
            test_true = np.empty((0))
            for batch in iterate_minibatches(X_test, argmax_y, BATCH_SIZE):
                inputs, targets = batch
                y_pred = sess.run(argmax_pred, feed_dict={xs: inputs, is_training: False})
                test_pred = np.append(test_pred, y_pred, axis=0)
                test_true = np.append(test_true, targets, axis=0)
            # pred = sess.run(argmax_pred, feed_dict={xs: X_test,  is_training: False})
            print('Step:', step, '| train loss: %.4f' % loss_
                  # , '| f1分数: %.4f' % (100 * metrics.f1_score(argmax_y, pred, average='weighted'))
                  , '| f1分数: %.4f' % (100 * metrics.f1_score(test_true, test_pred, average='weighted'))
                  , '| time: %.4f' % (time.time() - start_time))
        step += 1

# padding = same
# Step: 50 | train loss: 2.9848 | f1分数: 75.6440 | time: 3.2459
# Step: 100 | train loss: 1.1242 | f1分数: 75.6440 | time: 5.4250
# Step: 150 | train loss: 1.1085 | f1分数: 75.6440 | time: 7.6282
# Step: 200 | train loss: 0.7838 | f1分数: 75.6440 | time: 9.8296
# Step: 250 | train loss: 1.1174 | f1分数: 75.6440 | time: 12.0081
# Step: 300 | train loss: 2.0281 | f1分数: 76.9262 | time: 14.1963
# Step: 350 | train loss: 0.7262 | f1分数: 77.2497 | time: 16.4059
# Step: 400 | train loss: 1.3836 | f1分数: 77.0073 | time: 18.5541
# Step: 450 | train loss: 0.7132 | f1分数: 78.9395 | time: 20.7632
# Step: 500 | train loss: 1.2763 | f1分数: 77.7380 | time: 22.9590
# Step: 550 | train loss: 1.2018 | f1分数: 79.5135 | time: 25.1523
# Step: 600 | train loss: 0.4815 | f1分数: 79.5936 | time: 27.3460
# Step: 650 | train loss: 0.5322 | f1分数: 79.3532 | time: 29.5621
# Step: 700 | train loss: 0.9731 | f1分数: 79.8378 | time: 31.7750
# Step: 750 | train loss: 0.3670 | f1分数: 80.5998 | time: 33.9952
# Step: 800 | train loss: 0.6456 | f1分数: 80.2308 | time: 36.2051
# Step: 850 | train loss: 0.5758 | f1分数: 80.4721 | time: 38.3913
# Step: 900 | train loss: 0.6857 | f1分数: 80.7912 | time: 40.5686
# Step: 950 | train loss: 0.6460 | f1分数: 81.4541 | time: 42.7395
# Step: 1000 | train loss: 0.7116 | f1分数: 84.3133 | time: 44.9291
# Step: 1050 | train loss: 0.4917 | f1分数: 84.1999 | time: 47.1245
# Step: 1100 | train loss: 0.5265 | f1分数: 84.4007 | time: 49.3241
# Step: 1150 | train loss: 0.6587 | f1分数: 83.4359 | time: 51.5261
# Step: 1200 | train loss: 0.5761 | f1分数: 82.4672 | time: 53.7226
# Step: 1250 | train loss: 0.6244 | f1分数: 81.0575 | time: 55.8916
# Step: 1300 | train loss: 0.6922 | f1分数: 83.8709 | time: 58.1106
# Step: 1350 | train loss: 0.5909 | f1分数: 84.2277 | time: 60.2916
# Step: 1400 | train loss: 0.1513 | f1分数: 83.4572 | time: 62.4905
# Step: 1450 | train loss: 0.2916 | f1分数: 81.6557 | time: 64.7158
# Step: 1500 | train loss: 0.2445 | f1分数: 81.1641 | time: 66.9298
# Step: 1550 | train loss: 0.2679 | f1分数: 82.1488 | time: 69.1540
# Step: 1600 | train loss: 0.2279 | f1分数: 84.6383 | time: 71.3897
# Step: 1650 | train loss: 0.4817 | f1分数: 86.2233 | time: 73.5973
# Step: 1700 | train loss: 0.3180 | f1分数: 87.1407 | time: 75.8014
# Step: 1750 | train loss: 0.3163 | f1分数: 87.0351 | time: 77.9851
# Step: 1800 | train loss: 0.4101 | f1分数: 87.2125 | time: 80.1831
# Step: 1850 | train loss: 0.3539 | f1分数: 83.4132 | time: 82.4354
# Step: 1900 | train loss: 0.2586 | f1分数: 79.5796 | time: 84.6632
# Step: 1950 | train loss: 0.2715 | f1分数: 85.3532 | time: 86.8675
# Step: 2000 | train loss: 0.4713 | f1分数: 86.1622 | time: 89.0761
# Step: 2050 | train loss: 0.3869 | f1分数: 85.9695 | time: 91.2588
# Step: 2100 | train loss: 0.2980 | f1分数: 84.9429 | time: 93.4470
# Step: 2150 | train loss: 0.3672 | f1分数: 82.1115 | time: 95.6625
# Step: 2200 | train loss: 0.3707 | f1分数: 82.7529 | time: 97.8833
# Step: 2250 | train loss: 0.2340 | f1分数: 85.5584 | time: 100.1226
# Step: 2300 | train loss: 0.3552 | f1分数: 86.4594 | time: 102.3273
# Step: 2350 | train loss: 0.2058 | f1分数: 86.7493 | time: 104.5446
# Step: 2400 | train loss: 0.2876 | f1分数: 87.2978 | time: 106.7517
# Step: 2450 | train loss: 0.1827 | f1分数: 87.8640 | time: 108.9513
# Step: 2500 | train loss: 0.2774 | f1分数: 88.6876 | time: 111.1910
# Step: 2550 | train loss: 0.1229 | f1分数: 87.7341 | time: 113.4138
# Step: 2600 | train loss: 0.1613 | f1分数: 87.5300 | time: 115.6266
# Step: 2650 | train loss: 0.3649 | f1分数: 86.9785 | time: 117.8233
# Step: 2700 | train loss: 0.2914 | f1分数: 87.2052 | time: 120.0239
# Step: 2750 | train loss: 0.0787 | f1分数: 87.6544 | time: 122.2150
# Step: 2800 | train loss: 0.1511 | f1分数: 85.3612 | time: 124.4326
# Step: 2850 | train loss: 0.2792 | f1分数: 76.1038 | time: 126.6549
# Step: 2900 | train loss: 0.1871 | f1分数: 77.6485 | time: 128.8544
# Step: 2950 | train loss: 0.3862 | f1分数: 84.3305 | time: 131.0682
# Step: 3000 | train loss: 0.1352 | f1分数: 87.2343 | time: 133.2730
# Step: 3050 | train loss: 0.2722 | f1分数: 87.7788 | time: 135.4742
# Step: 3100 | train loss: 0.2641 | f1分数: 87.7255 | time: 137.6819
# Step: 3150 | train loss: 0.2575 | f1分数: 88.3656 | time: 139.9110
# Step: 3200 | train loss: 0.2041 | f1分数: 88.6307 | time: 142.0848
# Step: 3250 | train loss: 0.1760 | f1分数: 88.2297 | time: 144.3014
# Step: 3300 | train loss: 0.1319 | f1分数: 88.2800 | time: 146.5168
# Step: 3350 | train loss: 0.2645 | f1分数: 88.5382 | time: 148.7393
# Step: 3400 | train loss: 0.1789 | f1分数: 87.3511 | time: 150.9623
# Step: 3450 | train loss: 0.2290 | f1分数: 87.6070 | time: 153.1721
# Step: 3500 | train loss: 0.1771 | f1分数: 84.6178 | time: 155.3828
# Step: 3550 | train loss: 0.1889 | f1分数: 83.3187 | time: 157.6041
# Step: 3600 | train loss: 0.1233 | f1分数: 86.2042 | time: 159.7753
# Step: 3650 | train loss: 0.1648 | f1分数: 86.9671 | time: 161.9944
# Step: 3700 | train loss: 0.1589 | f1分数: 86.9312 | time: 164.1982
# Step: 3750 | train loss: 0.1650 | f1分数: 86.7030 | time: 166.3892
# Step: 3800 | train loss: 0.3064 | f1分数: 87.3981 | time: 168.5715
# Step: 3850 | train loss: 0.1489 | f1分数: 87.3399 | time: 170.7764
# Step: 3900 | train loss: 0.2168 | f1分数: 87.0128 | time: 172.9469
# Step: 3950 | train loss: 0.2007 | f1分数: 88.4736 | time: 175.1395
# Step: 4000 | train loss: 0.1993 | f1分数: 88.6525 | time: 177.3451
# Step: 4050 | train loss: 0.0644 | f1分数: 88.0752 | time: 179.5536
# Step: 4100 | train loss: 0.0913 | f1分数: 87.4779 | time: 181.7457
# Step: 4150 | train loss: 0.0930 | f1分数: 86.3124 | time: 183.9154
# Step: 4200 | train loss: 0.0915 | f1分数: 86.2024 | time: 186.1253
# Step: 4250 | train loss: 0.0776 | f1分数: 87.6060 | time: 188.3172
# Step: 4300 | train loss: 0.1328 | f1分数: 88.1369 | time: 190.5015
# Step: 4350 | train loss: 0.1007 | f1分数: 88.5341 | time: 192.6825
# Step: 4400 | train loss: 0.1016 | f1分数: 88.5237 | time: 194.8591
# Step: 4450 | train loss: 0.0502 | f1分数: 88.7366 | time: 197.0390
# Step: 4500 | train loss: 0.1288 | f1分数: 87.6308 | time: 199.2313
# Step: 4550 | train loss: 0.0705 | f1分数: 88.1031 | time: 201.4199
# Step: 4600 | train loss: 0.0749 | f1分数: 87.5752 | time: 203.6118
# Step: 4650 | train loss: 0.1380 | f1分数: 88.4621 | time: 205.8235
# Step: 4700 | train loss: 0.1335 | f1分数: 88.2362 | time: 208.0196
# Step: 4750 | train loss: 0.0780 | f1分数: 88.0832 | time: 210.1936
# Step: 4800 | train loss: 0.1184 | f1分数: 84.8186 | time: 212.3955
# Step: 4850 | train loss: 0.0535 | f1分数: 83.9676 | time: 214.6234
# Step: 4900 | train loss: 0.0563 | f1分数: 85.3750 | time: 216.8208
# Step: 4950 | train loss: 0.1105 | f1分数: 86.7004 | time: 219.0178
# Step: 5000 | train loss: 0.0691 | f1分数: 86.5304 | time: 221.2272
# Step: 5050 | train loss: 0.0472 | f1分数: 87.8588 | time: 223.4122
# Step: 5100 | train loss: 0.0836 | f1分数: 88.8770 | time: 225.6051
# Step: 5150 | train loss: 0.1280 | f1分数: 89.3812 | time: 227.8179
# Step: 5200 | train loss: 0.0571 | f1分数: 88.9991 | time: 230.0596
# Step: 5250 | train loss: 0.0267 | f1分数: 89.1275 | time: 232.2554
# Step: 5300 | train loss: 0.1227 | f1分数: 89.3686 | time: 234.4501
# Step: 5350 | train loss: 0.1174 | f1分数: 89.9180 | time: 236.6472
# Step: 5400 | train loss: 0.1081 | f1分数: 88.5223 | time: 238.8439
# Step: 5450 | train loss: 0.1120 | f1分数: 88.2835 | time: 241.0236
# Step: 5500 | train loss: 0.1033 | f1分数: 87.0675 | time: 243.2241
# Step: 5550 | train loss: 0.0937 | f1分数: 86.6171 | time: 245.4240
# Step: 5600 | train loss: 0.1248 | f1分数: 88.4989 | time: 247.6535
# Step: 5650 | train loss: 0.0391 | f1分数: 88.9387 | time: 249.8395
# Step: 5700 | train loss: 0.2852 | f1分数: 88.0929 | time: 252.0330
# Step: 5750 | train loss: 0.0456 | f1分数: 87.7766 | time: 254.2493
# Step: 5800 | train loss: 0.0893 | f1分数: 88.9567 | time: 256.4650
# Step: 5850 | train loss: 0.0656 | f1分数: 87.6921 | time: 258.6735
# Step: 5900 | train loss: 0.0733 | f1分数: 88.8130 | time: 260.8703
# Step: 5950 | train loss: 0.0196 | f1分数: 89.3729 | time: 263.0671
# Step: 6000 | train loss: 0.0824 | f1分数: 88.8773 | time: 265.2907
# Step: 6050 | train loss: 0.0624 | f1分数: 88.2991 | time: 267.4924
# Step: 6100 | train loss: 0.0438 | f1分数: 88.7951 | time: 269.7017
# Step: 6150 | train loss: 0.0509 | f1分数: 88.9256 | time: 271.9186
# Step: 6200 | train loss: 0.0341 | f1分数: 88.2619 | time: 274.1505
# Step: 6250 | train loss: 0.0072 | f1分数: 88.2858 | time: 276.3340
# Step: 6300 | train loss: 0.0426 | f1分数: 87.8071 | time: 278.5330
# Step: 6350 | train loss: 0.0282 | f1分数: 87.6238 | time: 280.7156
# Step: 6400 | train loss: 0.0485 | f1分数: 87.3079 | time: 282.9138
# Step: 6450 | train loss: 0.1329 | f1分数: 88.9235 | time: 285.1144
# Step: 6500 | train loss: 0.0681 | f1分数: 88.6986 | time: 287.3254
# Step: 6550 | train loss: 0.0695 | f1分数: 88.3387 | time: 289.4865
# Step: 6600 | train loss: 0.0674 | f1分数: 88.0118 | time: 291.7049
# Step: 6650 | train loss: 0.0932 | f1分数: 89.0912 | time: 293.8840
# Step: 6700 | train loss: 0.0157 | f1分数: 87.4408 | time: 296.1231
# Step: 6750 | train loss: 0.0261 | f1分数: 88.9550 | time: 298.3537
# Step: 6800 | train loss: 0.0940 | f1分数: 87.9120 | time: 300.5574
# Step: 6850 | train loss: 0.0197 | f1分数: 88.5782 | time: 302.7756
# Step: 6900 | train loss: 0.0280 | f1分数: 88.6075 | time: 304.9872
# Step: 6950 | train loss: 0.0369 | f1分数: 88.8000 | time: 307.1985
# Step: 7000 | train loss: 0.0163 | f1分数: 88.7921 | time: 309.3754
# Step: 7050 | train loss: 0.0238 | f1分数: 89.5039 | time: 311.5698
# Step: 7100 | train loss: 0.0036 | f1分数: 88.9367 | time: 313.8064
# Step: 7150 | train loss: 0.0541 | f1分数: 88.4402 | time: 316.0315
# Step: 7200 | train loss: 0.0120 | f1分数: 89.1289 | time: 318.2203
# Step: 7250 | train loss: 0.0068 | f1分数: 88.9972 | time: 320.4179
# Step: 7300 | train loss: 0.0457 | f1分数: 88.0440 | time: 322.6419
# Step: 7350 | train loss: 0.0574 | f1分数: 88.7058 | time: 324.8534
# Step: 7400 | train loss: 0.1221 | f1分数: 88.8566 | time: 327.0827
# Step: 7450 | train loss: 0.0792 | f1分数: 88.6430 | time: 329.2741
# Step: 7500 | train loss: 0.0100 | f1分数: 88.4628 | time: 331.4752
# Step: 7550 | train loss: 0.0196 | f1分数: 88.6737 | time: 333.6909
# Step: 7600 | train loss: 0.0742 | f1分数: 89.1906 | time: 335.8938
# Step: 7650 | train loss: 0.0114 | f1分数: 89.3210 | time: 338.0995
# Step: 7700 | train loss: 0.0218 | f1分数: 88.8746 | time: 340.2993
# Step: 7750 | train loss: 0.0490 | f1分数: 88.9226 | time: 342.4680
# Step: 7800 | train loss: 0.0587 | f1分数: 89.2423 | time: 344.6673
# Step: 7850 | train loss: 0.0528 | f1分数: 88.6991 | time: 346.8983
# Step: 7900 | train loss: 0.0166 | f1分数: 88.9534 | time: 349.1290
# Step: 7950 | train loss: 0.0512 | f1分数: 88.9663 | time: 351.3389
# Step: 8000 | train loss: 0.0188 | f1分数: 88.9380 | time: 353.5723
# Step: 8050 | train loss: 0.0174 | f1分数: 89.5737 | time: 355.8164
# Step: 8100 | train loss: 0.0417 | f1分数: 89.2239 | time: 358.0287
# Step: 8150 | train loss: 0.0331 | f1分数: 88.2342 | time: 360.2760
# Step: 8200 | train loss: 0.0616 | f1分数: 89.5282 | time: 362.4908
# Step: 8250 | train loss: 0.0256 | f1分数: 89.0009 | time: 364.7141
# Step: 8300 | train loss: 0.0175 | f1分数: 87.3828 | time: 366.9449
# Step: 8350 | train loss: 0.0571 | f1分数: 88.8265 | time: 369.1901
# Step: 8400 | train loss: 0.0149 | f1分数: 87.0435 | time: 371.4020
# Step: 8450 | train loss: 0.0683 | f1分数: 87.0671 | time: 373.6040
# Step: 8500 | train loss: 0.0451 | f1分数: 88.5277 | time: 375.7851
# Step: 8550 | train loss: 0.0570 | f1分数: 89.0321 | time: 378.0134
# Step: 8600 | train loss: 0.0104 | f1分数: 88.6298 | time: 380.2279
# Step: 8650 | train loss: 0.0380 | f1分数: 88.7494 | time: 382.4533
# Step: 8700 | train loss: 0.0005 | f1分数: 89.5746 | time: 384.6822
# Step: 8750 | train loss: 0.0124 | f1分数: 86.8014 | time: 386.9027
# Step: 8800 | train loss: 0.0127 | f1分数: 88.5307 | time: 389.1849
# Step: 8850 | train loss: 0.0167 | f1分数: 89.0914 | time: 391.3774
# Step: 8900 | train loss: 0.0018 | f1分数: 88.9217 | time: 393.5882
# Step: 8950 | train loss: 0.0324 | f1分数: 89.1565 | time: 395.7999
# Step: 9000 | train loss: 0.0048 | f1分数: 89.0748 | time: 397.9922
# Step: 9050 | train loss: 0.0190 | f1分数: 88.4327 | time: 400.2041
# Step: 9100 | train loss: 0.0437 | f1分数: 88.9143 | time: 402.4270
# Step: 9150 | train loss: 0.0794 | f1分数: 89.1531 | time: 404.6187
# Step: 9200 | train loss: 0.0429 | f1分数: 87.4195 | time: 406.8342
# Step: 9250 | train loss: 0.1062 | f1分数: 88.7150 | time: 409.0514
# Step: 9300 | train loss: 0.0546 | f1分数: 89.1973 | time: 411.2561
# Step: 9350 | train loss: 0.0203 | f1分数: 89.2418 | time: 413.4810
# Step: 9400 | train loss: 0.0084 | f1分数: 89.0577 | time: 415.6976
# Step: 9450 | train loss: 0.0312 | f1分数: 89.2037 | time: 417.8945
# Step: 9500 | train loss: 0.0048 | f1分数: 88.9991 | time: 420.1230
# Step: 9550 | train loss: 0.0083 | f1分数: 89.4811 | time: 422.3599
# Step: 9600 | train loss: 0.0159 | f1分数: 88.3025 | time: 424.5610
# Step: 9650 | train loss: 0.0096 | f1分数: 85.8411 | time: 426.7871
# Step: 9700 | train loss: 0.0211 | f1分数: 86.3881 | time: 428.9870
# Step: 9750 | train loss: 0.0013 | f1分数: 89.2064 | time: 431.1976
# Step: 9800 | train loss: 0.0233 | f1分数: 89.3070 | time: 433.4123
# Step: 9850 | train loss: 0.0067 | f1分数: 89.2549 | time: 435.6198
# Step: 9900 | train loss: 0.0023 | f1分数: 87.6644 | time: 437.8277
# Step: 9950 | train loss: 0.0174 | f1分数: 89.0411 | time: 440.0430
# Step: 10000 | train loss: 0.0087 | f1分数: 89.0411 | time: 442.2668
# Step: 10050 | train loss: 0.0039 | f1分数: 89.3312 | time: 444.4840
# Step: 10100 | train loss: 0.0094 | f1分数: 89.5744 | time: 446.6981
# Step: 10150 | train loss: 0.0103 | f1分数: 88.8962 | time: 448.8936
# Step: 10200 | train loss: 0.0095 | f1分数: 89.0387 | time: 451.1035
# Step: 10250 | train loss: 0.0304 | f1分数: 88.0447 | time: 453.3083
# Step: 10300 | train loss: 0.0041 | f1分数: 89.4100 | time: 455.5506
# Step: 10350 | train loss: 0.0615 | f1分数: 89.1278 | time: 457.7383
# Step: 10400 | train loss: 0.0154 | f1分数: 88.3184 | time: 459.9553
# Step: 10450 | train loss: 0.0534 | f1分数: 88.9718 | time: 462.1513
# Step: 10500 | train loss: 0.0278 | f1分数: 88.7091 | time: 464.3678
# Step: 10550 | train loss: 0.0073 | f1分数: 87.2817 | time: 466.5818
# Step: 10600 | train loss: 0.0030 | f1分数: 89.4159 | time: 468.7809
# Step: 10650 | train loss: 0.0233 | f1分数: 89.0890 | time: 471.0097
# Step: 10700 | train loss: 0.0195 | f1分数: 86.7087 | time: 473.1805
# Step: 10750 | train loss: 0.0064 | f1分数: 89.1828 | time: 475.3966
# Step: 10800 | train loss: 0.0275 | f1分数: 89.3904 | time: 477.5772
# Step: 10850 | train loss: 0.0560 | f1分数: 89.1883 | time: 479.7479
# Step: 10900 | train loss: 0.0180 | f1分数: 89.2398 | time: 481.9234
# Step: 10950 | train loss: 0.0095 | f1分数: 89.3413 | time: 484.1309
# Step: 11000 | train loss: 0.0224 | f1分数: 88.9562 | time: 486.3311
# Step: 11050 | train loss: 0.0198 | f1分数: 89.1191 | time: 488.5812
# Step: 11100 | train loss: 0.0405 | f1分数: 89.3234 | time: 490.7944
# Step: 11150 | train loss: 0.0108 | f1分数: 89.4513 | time: 493.0339
# Step: 11200 | train loss: 0.0184 | f1分数: 88.1503 | time: 495.2425
# Step: 11250 | train loss: 0.0073 | f1分数: 89.4255 | time: 497.4511
# Step: 11300 | train loss: 0.0098 | f1分数: 89.1159 | time: 499.6681
# Step: 11350 | train loss: 0.0004 | f1分数: 89.1308 | time: 501.8697
# Step: 11400 | train loss: 0.0103 | f1分数: 88.8710 | time: 504.0685
# Step: 11450 | train loss: 0.0231 | f1分数: 88.6802 | time: 506.2852
# Step: 11500 | train loss: 0.0096 | f1分数: 89.0878 | time: 508.5129
# Step: 11550 | train loss: 0.0070 | f1分数: 89.2080 | time: 510.6935
# Step: 11600 | train loss: 0.0150 | f1分数: 87.8636 | time: 512.9078
# Step: 11650 | train loss: 0.0030 | f1分数: 89.1464 | time: 515.1241
# Step: 11700 | train loss: 0.0013 | f1分数: 89.2020 | time: 517.3426
# Step: 11750 | train loss: 0.0054 | f1分数: 89.4314 | time: 519.5736
# Step: 11800 | train loss: 0.0054 | f1分数: 89.3313 | time: 521.7713
# Step: 11850 | train loss: 0.0021 | f1分数: 89.4843 | time: 523.9897
# Step: 11900 | train loss: 0.0074 | f1分数: 89.5364 | time: 526.2055
# Step: 11950 | train loss: 0.0098 | f1分数: 89.4895 | time: 528.4274
# Step: 12000 | train loss: 0.0039 | f1分数: 88.5347 | time: 530.6544
# Step: 12050 | train loss: 0.1617 | f1分数: 85.7723 | time: 532.8741
# Step: 12100 | train loss: 0.0695 | f1分数: 88.9367 | time: 535.0921
# Step: 12150 | train loss: 0.0616 | f1分数: 88.4691 | time: 537.3201
# Step: 12200 | train loss: 0.0272 | f1分数: 89.4201 | time: 539.5097
# Step: 12250 | train loss: 0.0117 | f1分数: 89.0492 | time: 541.6918
# Step: 12300 | train loss: 0.0086 | f1分数: 88.3538 | time: 543.9110
# Step: 12350 | train loss: 0.0065 | f1分数: 89.2318 | time: 546.1253
# Step: 12400 | train loss: 0.0022 | f1分数: 89.5046 | time: 548.3363
# Step: 12450 | train loss: 0.0124 | f1分数: 89.6360 | time: 550.5648
# Step: 12500 | train loss: 0.0029 | f1分数: 89.5744 | time: 552.7702
# Step: 12550 | train loss: 0.0031 | f1分数: 89.0624 | time: 554.9862
# Step: 12600 | train loss: 0.0072 | f1分数: 89.9543 | time: 557.2088
# Step: 12650 | train loss: 0.0030 | f1分数: 89.0609 | time: 559.4492
# Step: 12700 | train loss: 0.0045 | f1分数: 89.4779 | time: 561.6642
# Step: 12750 | train loss: 0.0084 | f1分数: 88.8238 | time: 563.8793
# Step: 12800 | train loss: 0.0033 | f1分数: 88.0099 | time: 566.0927
# Step: 12850 | train loss: 0.0131 | f1分数: 88.1666 | time: 568.2720
# Step: 12900 | train loss: 0.0299 | f1分数: 87.3227 | time: 570.4925
# Step: 12950 | train loss: 0.0027 | f1分数: 88.4016 | time: 572.7127
# Step: 13000 | train loss: 0.0313 | f1分数: 88.6131 | time: 574.9289
# Step: 13050 | train loss: 0.0175 | f1分数: 88.5930 | time: 577.1609
# Step: 13100 | train loss: 0.0375 | f1分数: 88.6663 | time: 579.3695
# Step: 13150 | train loss: 0.0251 | f1分数: 88.9047 | time: 581.5914
# Step: 13200 | train loss: 0.0090 | f1分数: 89.4846 | time: 583.8163
# Step: 13250 | train loss: 0.0076 | f1分数: 89.5160 | time: 586.0222
# Step: 13300 | train loss: 0.0104 | f1分数: 89.7415 | time: 588.2232
# Step: 13350 | train loss: 0.0362 | f1分数: 89.0581 | time: 590.4312
# Step: 13400 | train loss: 0.0010 | f1分数: 89.7131 | time: 592.6307
# Step: 13450 | train loss: 0.0666 | f1分数: 89.0030 | time: 594.7979
# Step: 13500 | train loss: 0.0753 | f1分数: 85.7326 | time: 596.9568
# Step: 13550 | train loss: 0.0106 | f1分数: 87.9898 | time: 599.1437
# Step: 13600 | train loss: 0.0059 | f1分数: 89.2033 | time: 601.3310
# Step: 13650 | train loss: 0.0146 | f1分数: 89.4455 | time: 603.4853
# Step: 13700 | train loss: 0.0013 | f1分数: 89.0331 | time: 605.6962
# Step: 13750 | train loss: 0.0725 | f1分数: 89.4818 | time: 607.9249
# Step: 13800 | train loss: 0.0036 | f1分数: 89.4980 | time: 610.0954
# Step: 13850 | train loss: 0.0082 | f1分数: 89.4792 | time: 612.2597
# Step: 13900 | train loss: 0.0051 | f1分数: 88.6367 | time: 614.4595

# kernel_size = (1, 8)
# Step: 50 | train loss: 3.1211 | f1分数: 75.6440 | time: 4.8672
# Step: 100 | train loss: 1.1815 | f1分数: 75.6440 | time: 8.4192
# Step: 150 | train loss: 0.9684 | f1分数: 75.6440 | time: 12.0285
# Step: 200 | train loss: 0.7005 | f1分数: 75.6440 | time: 15.6324
# Step: 250 | train loss: 0.8409 | f1分数: 76.0807 | time: 19.2542
# Step: 300 | train loss: 1.8033 | f1分数: 78.6574 | time: 22.8759
# Step: 350 | train loss: 0.7046 | f1分数: 75.8225 | time: 26.4818
# Step: 400 | train loss: 1.1453 | f1分数: 80.5313 | time: 30.1309
# Step: 450 | train loss: 0.6190 | f1分数: 81.5831 | time: 33.7141
# Step: 500 | train loss: 0.8124 | f1分数: 82.2451 | time: 37.3024
# Step: 550 | train loss: 0.7579 | f1分数: 82.0269 | time: 40.9030
# Step: 600 | train loss: 0.3263 | f1分数: 82.4667 | time: 44.4960
# Step: 650 | train loss: 0.4027 | f1分数: 82.5880 | time: 48.1264
# Step: 700 | train loss: 0.5796 | f1分数: 84.1146 | time: 51.7298
# Step: 750 | train loss: 0.2802 | f1分数: 84.8383 | time: 55.3738
# Step: 800 | train loss: 0.3962 | f1分数: 84.3047 | time: 58.9943
# Step: 850 | train loss: 0.4155 | f1分数: 83.7371 | time: 62.6250
# Step: 900 | train loss: 0.3834 | f1分数: 83.4530 | time: 66.2221
# Step: 950 | train loss: 0.4137 | f1分数: 84.4687 | time: 69.8134
# Step: 1000 | train loss: 0.3737 | f1分数: 85.2086 | time: 73.4091
# Step: 1050 | train loss: 0.3565 | f1分数: 85.0611 | time: 77.0202
# Step: 1100 | train loss: 0.3661 | f1分数: 87.3096 | time: 80.6202
# Step: 1150 | train loss: 0.4497 | f1分数: 88.2415 | time: 84.2587
# Step: 1200 | train loss: 0.3591 | f1分数: 86.6073 | time: 87.8842
# Step: 1250 | train loss: 0.4057 | f1分数: 87.2114 | time: 91.4850
# Step: 1300 | train loss: 0.4484 | f1分数: 86.5808 | time: 95.0972
# Step: 1350 | train loss: 0.4041 | f1分数: 87.1940 | time: 98.7189
# Step: 1400 | train loss: 0.0760 | f1分数: 86.7134 | time: 102.3383
# Step: 1450 | train loss: 0.1648 | f1分数: 85.2376 | time: 105.9545
# Step: 1500 | train loss: 0.1465 | f1分数: 83.5057 | time: 109.5959
# Step: 1550 | train loss: 0.1401 | f1分数: 82.8816 | time: 113.1704
# Step: 1600 | train loss: 0.1629 | f1分数: 83.7007 | time: 116.7637
# Step: 1650 | train loss: 0.2242 | f1分数: 84.7260 | time: 120.4151
# Step: 1700 | train loss: 0.2290 | f1分数: 85.1503 | time: 124.0208
# Step: 1750 | train loss: 0.1859 | f1分数: 86.1719 | time: 127.6572
# Step: 1800 | train loss: 0.2839 | f1分数: 86.4541 | time: 131.3161
# Step: 1850 | train loss: 0.2172 | f1分数: 87.5322 | time: 134.9368
# Step: 1900 | train loss: 0.1644 | f1分数: 87.3753 | time: 138.5691
# Step: 1950 | train loss: 0.1989 | f1分数: 87.8302 | time: 142.1458
# Step: 2000 | train loss: 0.2791 | f1分数: 88.4334 | time: 145.7825
# Step: 2050 | train loss: 0.2249 | f1分数: 87.8049 | time: 149.4104
# Step: 2100 | train loss: 0.1668 | f1分数: 86.7883 | time: 152.9664
# Step: 2150 | train loss: 0.1959 | f1分数: 87.4115 | time: 156.5724
# Step: 2200 | train loss: 0.1174 | f1分数: 87.5828 | time: 160.2222
# Step: 2250 | train loss: 0.1108 | f1分数: 86.9830 | time: 163.8702
# Step: 2300 | train loss: 0.2504 | f1分数: 86.9215 | time: 167.5008
# Step: 2350 | train loss: 0.0797 | f1分数: 88.1125 | time: 171.1330
# Step: 2400 | train loss: 0.1088 | f1分数: 88.6189 | time: 174.8129
# Step: 2450 | train loss: 0.0594 | f1分数: 88.9600 | time: 178.4261
# Step: 2500 | train loss: 0.1631 | f1分数: 89.2076 | time: 182.0518
# Step: 2550 | train loss: 0.0798 | f1分数: 82.2916 | time: 185.6687
# Step: 2600 | train loss: 0.0792 | f1分数: 89.4384 | time: 189.2712
# Step: 2650 | train loss: 0.1566 | f1分数: 89.6598 | time: 192.8919
# Step: 2700 | train loss: 0.1436 | f1分数: 89.8685 | time: 196.5004
# Step: 2750 | train loss: 0.0560 | f1分数: 87.4650 | time: 200.0998
# Step: 2800 | train loss: 0.0887 | f1分数: 87.7890 | time: 203.6993
# Step: 2850 | train loss: 0.1041 | f1分数: 87.5832 | time: 207.3079
# Step: 2900 | train loss: 0.1249 | f1分数: 85.8584 | time: 210.9330
# Step: 2950 | train loss: 0.2083 | f1分数: 87.0300 | time: 214.5490
# Step: 3000 | train loss: 0.0362 | f1分数: 87.8871 | time: 218.1365
# Step: 3050 | train loss: 0.1337 | f1分数: 87.9839 | time: 221.7201
# Step: 3100 | train loss: 0.1037 | f1分数: 87.8109 | time: 225.3309
# Step: 3150 | train loss: 0.0960 | f1分数: 87.3107 | time: 228.9298
# Step: 3200 | train loss: 0.0986 | f1分数: 82.3016 | time: 232.5314
# Step: 3250 | train loss: 0.0698 | f1分数: 88.5992 | time: 236.1676
# Step: 3300 | train loss: 0.0555 | f1分数: 88.4382 | time: 239.8017
# Step: 3350 | train loss: 0.1051 | f1分数: 88.6239 | time: 243.3838
# Step: 3400 | train loss: 0.0597 | f1分数: 88.8322 | time: 247.0035
# Step: 3450 | train loss: 0.0750 | f1分数: 87.8096 | time: 250.6230
# Step: 3500 | train loss: 0.0514 | f1分数: 87.9819 | time: 254.2169
# Step: 3550 | train loss: 0.1031 | f1分数: 88.6764 | time: 257.8284
# Step: 3600 | train loss: 0.0319 | f1分数: 89.0418 | time: 261.4322
# Step: 3650 | train loss: 0.0723 | f1分数: 88.9436 | time: 265.0433
# Step: 3700 | train loss: 0.0263 | f1分数: 88.1631 | time: 268.6918
# Step: 3750 | train loss: 0.0805 | f1分数: 88.3567 | time: 272.3141
# Step: 3800 | train loss: 0.2689 | f1分数: 88.6514 | time: 275.9272
# Step: 3850 | train loss: 0.0500 | f1分数: 89.0991 | time: 279.5410
# Step: 3900 | train loss: 0.0543 | f1分数: 88.8063 | time: 283.1393
# Step: 3950 | train loss: 0.0398 | f1分数: 88.3608 | time: 286.7730
# Step: 4000 | train loss: 0.0543 | f1分数: 87.9244 | time: 290.4024
# Step: 4050 | train loss: 0.0138 | f1分数: 88.3498 | time: 294.0139
# Step: 4100 | train loss: 0.0133 | f1分数: 86.6153 | time: 297.6597
# Step: 4150 | train loss: 0.0614 | f1分数: 87.3134 | time: 301.2878
# Step: 4200 | train loss: 0.0250 | f1分数: 89.2799 | time: 304.9173
# Step: 4250 | train loss: 0.0445 | f1分数: 89.7228 | time: 308.5399
# Step: 4300 | train loss: 0.0451 | f1分数: 89.2209 | time: 312.1491
# Step: 4350 | train loss: 0.0360 | f1分数: 88.0262 | time: 315.7560
# Step: 4400 | train loss: 0.0402 | f1分数: 88.9547 | time: 319.3767
# Step: 4450 | train loss: 0.0165 | f1分数: 88.9694 | time: 323.0028
# Step: 4500 | train loss: 0.0513 | f1分数: 88.9987 | time: 326.6164
# Step: 4550 | train loss: 0.0196 | f1分数: 88.4292 | time: 330.2838
# Step: 4600 | train loss: 0.0266 | f1分数: 89.5050 | time: 333.8868
# Step: 4650 | train loss: 0.0402 | f1分数: 88.7712 | time: 337.5225
# Step: 4700 | train loss: 0.0368 | f1分数: 88.6311 | time: 341.1458
# Step: 4750 | train loss: 0.0256 | f1分数: 88.9266 | time: 344.7608
# Step: 4800 | train loss: 0.1520 | f1分数: 89.0422 | time: 348.3674
# Step: 4850 | train loss: 0.0101 | f1分数: 89.4722 | time: 352.0231
# Step: 4900 | train loss: 0.0153 | f1分数: 87.5879 | time: 355.6216
# Step: 4950 | train loss: 0.0664 | f1分数: 88.8315 | time: 359.2405
# Step: 5000 | train loss: 0.0131 | f1分数: 88.0868 | time: 362.8522
# Step: 5050 | train loss: 0.0121 | f1分数: 89.1292 | time: 366.4832
# Step: 5100 | train loss: 0.0224 | f1分数: 88.6062 | time: 370.1088
# Step: 5150 | train loss: 0.0573 | f1分数: 88.5966 | time: 373.7285
# Step: 5200 | train loss: 0.0239 | f1分数: 88.3113 | time: 377.3501
# Step: 5250 | train loss: 0.0099 | f1分数: 88.5151 | time: 380.9722
# Step: 5300 | train loss: 0.2430 | f1分数: 84.4189 | time: 384.5755
# Step: 5350 | train loss: 0.0310 | f1分数: 88.4012 | time: 388.1815
# Step: 5400 | train loss: 0.0481 | f1分数: 88.7057 | time: 391.8311
# Step: 5450 | train loss: 0.0206 | f1分数: 87.5186 | time: 395.4515
# Step: 5500 | train loss: 0.0240 | f1分数: 87.8707 | time: 399.0980
# Step: 5550 | train loss: 0.0791 | f1分数: 88.9442 | time: 402.7133
# Step: 5600 | train loss: 0.0742 | f1分数: 88.4342 | time: 406.3324
# Step: 5650 | train loss: 0.0114 | f1分数: 87.8612 | time: 409.9192
# Step: 5700 | train loss: 0.1042 | f1分数: 89.0152 | time: 413.5156
# Step: 5750 | train loss: 0.0082 | f1分数: 88.8095 | time: 417.1352
# Step: 5800 | train loss: 0.0549 | f1分数: 89.0911 | time: 420.7337
# Step: 5850 | train loss: 0.0368 | f1分数: 89.1132 | time: 424.3732
# Step: 5900 | train loss: 0.0102 | f1分数: 88.4413 | time: 427.9781
# Step: 5950 | train loss: 0.0142 | f1分数: 88.1562 | time: 431.5763
# Step: 6000 | train loss: 0.0386 | f1分数: 88.5315 | time: 435.1973
# Step: 6050 | train loss: 0.0110 | f1分数: 88.7688 | time: 438.8209
# Step: 6100 | train loss: 0.0070 | f1分数: 88.2015 | time: 442.4145
# Step: 6150 | train loss: 0.0306 | f1分数: 88.7174 | time: 446.0252
# Step: 6200 | train loss: 0.0094 | f1分数: 89.1993 | time: 449.6476
# Step: 6250 | train loss: 0.0023 | f1分数: 88.8190 | time: 453.2605
# Step: 6300 | train loss: 0.0228 | f1分数: 87.8824 | time: 456.8682
# Step: 6350 | train loss: 0.0220 | f1分数: 88.7649 | time: 460.4758
# Step: 6400 | train loss: 0.0090 | f1分数: 87.7240 | time: 464.0782
# Step: 6450 | train loss: 0.1645 | f1分数: 88.5928 | time: 467.6884
# Step: 6500 | train loss: 0.0216 | f1分数: 88.9664 | time: 471.3240
# Step: 6550 | train loss: 0.0252 | f1分数: 88.6323 | time: 474.9213
# Step: 6600 | train loss: 0.0058 | f1分数: 88.6363 | time: 478.5297
# Step: 6650 | train loss: 0.0179 | f1分数: 87.8994 | time: 482.1319
# Step: 6700 | train loss: 0.0112 | f1分数: 87.0708 | time: 485.7172
# Step: 6750 | train loss: 0.0461 | f1分数: 87.6227 | time: 489.3319
# Step: 6800 | train loss: 0.0307 | f1分数: 89.4433 | time: 492.9464
# Step: 6850 | train loss: 0.0064 | f1分数: 88.7706 | time: 496.5600
# Step: 6900 | train loss: 0.0143 | f1分数: 89.1372 | time: 500.1988
# Step: 6950 | train loss: 0.0221 | f1分数: 88.8917 | time: 503.8135
# Step: 7000 | train loss: 0.0035 | f1分数: 88.6363 | time: 507.3857
# Step: 7050 | train loss: 0.0021 | f1分数: 89.4439 | time: 511.0208
# Step: 7100 | train loss: 0.0029 | f1分数: 89.3723 | time: 514.6368
# Step: 7150 | train loss: 0.0406 | f1分数: 88.0363 | time: 518.2643
# Step: 7200 | train loss: 0.0072 | f1分数: 88.0174 | time: 521.8972
# Step: 7250 | train loss: 0.0040 | f1分数: 88.9417 | time: 525.5070
# Step: 7300 | train loss: 0.0167 | f1分数: 88.3643 | time: 529.1082
# Step: 7350 | train loss: 0.0141 | f1分数: 89.0440 | time: 532.7288
# Step: 7400 | train loss: 0.0261 | f1分数: 88.6714 | time: 536.3524
# Step: 7450 | train loss: 0.0686 | f1分数: 88.2547 | time: 539.9402
# Step: 7500 | train loss: 0.0074 | f1分数: 88.3815 | time: 543.5039
# Step: 7550 | train loss: 0.0044 | f1分数: 88.6346 | time: 547.1082
# Step: 7600 | train loss: 0.0309 | f1分数: 88.1510 | time: 550.7516
# Step: 7650 | train loss: 0.0028 | f1分数: 89.5254 | time: 554.3530
# Step: 7700 | train loss: 0.0132 | f1分数: 89.6287 | time: 557.9616
# Step: 7750 | train loss: 0.0091 | f1分数: 89.3389 | time: 561.5641
# Step: 7800 | train loss: 0.0356 | f1分数: 88.6093 | time: 565.1532
# Step: 7850 | train loss: 0.0293 | f1分数: 89.2626 | time: 568.7249
# Step: 7900 | train loss: 0.0062 | f1分数: 88.8342 | time: 572.3338
# Step: 7950 | train loss: 0.0131 | f1分数: 89.0314 | time: 575.9507
# Step: 8000 | train loss: 0.0312 | f1分数: 88.9603 | time: 579.5585
# Step: 8050 | train loss: 0.0273 | f1分数: 88.8393 | time: 583.1902
# Step: 8100 | train loss: 0.0132 | f1分数: 88.5790 | time: 586.8378
# Step: 8150 | train loss: 0.0178 | f1分数: 89.3100 | time: 590.4458
# Step: 8200 | train loss: 0.0209 | f1分数: 87.9468 | time: 594.0283
# Step: 8250 | train loss: 0.0098 | f1分数: 88.6684 | time: 597.6682
# Step: 8300 | train loss: 0.0097 | f1分数: 88.5504 | time: 601.2541
# Step: 8350 | train loss: 0.0144 | f1分数: 88.9538 | time: 604.8483
# Step: 8400 | train loss: 0.0025 | f1分数: 89.1539 | time: 608.4651
# Step: 8450 | train loss: 0.0501 | f1分数: 89.2224 | time: 612.0563
# Step: 8500 | train loss: 0.0164 | f1分数: 89.0575 | time: 615.6712
# Step: 8550 | train loss: 0.0067 | f1分数: 88.7543 | time: 619.2926
# Step: 8600 | train loss: 0.0002 | f1分数: 89.6753 | time: 622.9354
# Step: 8650 | train loss: 0.0021 | f1分数: 88.3910 | time: 626.5568
# Step: 8700 | train loss: 0.0000 | f1分数: 89.3484 | time: 630.1804
# Step: 8750 | train loss: 0.0002 | f1分数: 88.9013 | time: 633.8030
# Step: 8800 | train loss: 0.0033 | f1分数: 88.9107 | time: 637.4302
# Step: 8850 | train loss: 0.0080 | f1分数: 89.6162 | time: 641.0146
# Step: 8900 | train loss: 0.0031 | f1分数: 88.8487 | time: 644.5966
# Step: 8950 | train loss: 0.0063 | f1分数: 88.5898 | time: 648.2460
# Step: 9000 | train loss: 0.0104 | f1分数: 87.3710 | time: 651.8747
# Step: 9050 | train loss: 0.5044 | f1分数: 88.8314 | time: 655.5006
# Step: 9100 | train loss: 0.0424 | f1分数: 89.3208 | time: 659.1292
# Step: 9150 | train loss: 0.0341 | f1分数: 88.3330 | time: 662.7320
# Step: 9200 | train loss: 0.0323 | f1分数: 87.6674 | time: 666.3479
# Step: 9250 | train loss: 0.0265 | f1分数: 88.1569 | time: 669.9468
# Step: 9300 | train loss: 0.0920 | f1分数: 88.4830 | time: 673.5647
# Step: 9350 | train loss: 0.0100 | f1分数: 87.8945 | time: 677.2010
# Step: 9400 | train loss: 0.0074 | f1分数: 88.2828 | time: 680.8151
# Step: 9450 | train loss: 0.0223 | f1分数: 88.8949 | time: 684.4111
# Step: 9500 | train loss: 0.0064 | f1分数: 88.7334 | time: 688.0603
# Step: 9550 | train loss: 0.0045 | f1分数: 89.7532 | time: 691.6776
# Step: 9600 | train loss: 0.0107 | f1分数: 89.7471 | time: 695.2636
# Step: 9650 | train loss: 0.0023 | f1分数: 89.7215 | time: 698.8695
# Step: 9700 | train loss: 0.0011 | f1分数: 89.6157 | time: 702.4976
# Step: 9750 | train loss: 0.0005 | f1分数: 89.2226 | time: 706.1600
# Step: 9800 | train loss: 0.0100 | f1分数: 89.4680 | time: 709.7945
# Step: 9850 | train loss: 0.0012 | f1分数: 89.3940 | time: 713.4019
# Step: 9900 | train loss: 0.0026 | f1分数: 89.4422 | time: 716.9897
# Step: 9950 | train loss: 0.0031 | f1分数: 89.2698 | time: 720.5619
# Step: 10000 | train loss: 0.0020 | f1分数: 88.8869 | time: 724.1824
# Step: 10050 | train loss: 0.0037 | f1分数: 88.9219 | time: 727.7961
# Step: 10100 | train loss: 0.0040 | f1分数: 88.8812 | time: 731.4088
# Step: 10150 | train loss: 0.0006 | f1分数: 89.0575 | time: 735.0019
# Step: 10200 | train loss: 0.0025 | f1分数: 89.0733 | time: 738.6158
# Step: 10250 | train loss: 0.0224 | f1分数: 89.5850 | time: 742.2373
# Step: 10300 | train loss: 0.0074 | f1分数: 89.2558 | time: 745.8557
# Step: 10350 | train loss: 0.0091 | f1分数: 89.0852 | time: 749.4581
# Step: 10400 | train loss: 0.0052 | f1分数: 89.4522 | time: 753.0824
# Step: 10450 | train loss: 0.0135 | f1分数: 89.6557 | time: 756.6918
# Step: 10500 | train loss: 0.0094 | f1分数: 89.3651 | time: 760.3085
# Step: 10550 | train loss: 0.0014 | f1分数: 89.2578 | time: 763.9273
# Step: 10600 | train loss: 0.0171 | f1分数: 87.4781 | time: 767.5441
# Step: 10650 | train loss: 0.0172 | f1分数: 87.8859 | time: 771.1583
# Step: 10700 | train loss: 0.0087 | f1分数: 85.9796 | time: 774.7828
# Step: 10750 | train loss: 0.0025 | f1分数: 85.7842 | time: 778.3919
# Step: 10800 | train loss: 0.0444 | f1分数: 88.4468 | time: 782.0012
# Step: 10850 | train loss: 0.1250 | f1分数: 88.0142 | time: 785.6241
# Step: 10900 | train loss: 0.0026 | f1分数: 88.3261 | time: 789.2134
# Step: 10950 | train loss: 0.0044 | f1分数: 88.9583 | time: 792.8190
# Step: 11000 | train loss: 0.2336 | f1分数: 89.1912 | time: 796.4150
# Step: 11050 | train loss: 0.0081 | f1分数: 89.4468 | time: 800.0619
# Step: 11100 | train loss: 0.0409 | f1分数: 89.3574 | time: 803.6533
# Step: 11150 | train loss: 0.0128 | f1分数: 89.6730 | time: 807.2183
# Step: 11200 | train loss: 0.0055 | f1分数: 89.4711 | time: 810.8181
# Step: 11250 | train loss: 0.0051 | f1分数: 89.2462 | time: 814.4346
# Step: 11300 | train loss: 0.0144 | f1分数: 89.2516 | time: 818.0450
# Step: 11350 | train loss: 0.0003 | f1分数: 89.4224 | time: 821.6741
# Step: 11400 | train loss: 0.0001 | f1分数: 88.8881 | time: 825.2937
# Step: 11450 | train loss: 0.0006 | f1分数: 89.3523 | time: 828.9331
# Step: 11500 | train loss: 0.0059 | f1分数: 88.3953 | time: 832.5434
# Step: 11550 | train loss: 0.0011 | f1分数: 89.7519 | time: 836.1626
# Step: 11600 | train loss: 0.0008 | f1分数: 89.4281 | time: 839.7696
# Step: 11650 | train loss: 0.0005 | f1分数: 89.3094 | time: 843.3597
# Step: 11700 | train loss: 0.0014 | f1分数: 89.2398 | time: 846.9796
# Step: 11750 | train loss: 0.0006 | f1分数: 89.2487 | time: 850.6192
# Step: 11800 | train loss: 0.0042 | f1分数: 89.2519 | time: 854.2277
# Step: 11850 | train loss: 0.0006 | f1分数: 88.9686 | time: 857.8513
# Step: 11900 | train loss: 0.0005 | f1分数: 89.3657 | time: 861.4556
# Step: 11950 | train loss: 0.0187 | f1分数: 88.9073 | time: 865.0534
# Step: 12000 | train loss: 0.0043 | f1分数: 89.7600 | time: 868.6642
# Step: 12050 | train loss: 0.0010 | f1分数: 89.0913 | time: 872.2819
# Step: 12100 | train loss: 0.0322 | f1分数: 89.4827 | time: 875.9124
# Step: 12150 | train loss: 0.0008 | f1分数: 89.0444 | time: 879.5348
# Step: 12200 | train loss: 0.0045 | f1分数: 88.9031 | time: 883.1319
# Step: 12250 | train loss: 0.0039 | f1分数: 88.7208 | time: 886.7236
# Step: 12300 | train loss: 0.0034 | f1分数: 89.3595 | time: 890.3374
# Step: 12350 | train loss: 0.0031 | f1分数: 88.4687 | time: 893.9310
# Step: 12400 | train loss: 0.0181 | f1分数: 89.6331 | time: 897.5644
# Step: 12450 | train loss: 0.0121 | f1分数: 89.2598 | time: 901.1465
# Step: 12500 | train loss: 0.0043 | f1分数: 89.0716 | time: 904.7592
# Step: 12550 | train loss: 0.0183 | f1分数: 88.6552 | time: 908.3783
# Step: 12600 | train loss: 0.0166 | f1分数: 88.7404 | time: 911.9910
# Step: 12650 | train loss: 0.0045 | f1分数: 89.0614 | time: 915.5905
# Step: 12700 | train loss: 0.0039 | f1分数: 88.6312 | time: 919.1916
# Step: 12750 | train loss: 0.0020 | f1分数: 89.3195 | time: 922.8054
# Step: 12800 | train loss: 0.0006 | f1分数: 89.6304 | time: 926.4221
# Step: 12850 | train loss: 0.0013 | f1分数: 89.3458 | time: 930.0638
# Step: 12900 | train loss: 0.0007 | f1分数: 89.5881 | time: 933.6530
# Step: 12950 | train loss: 0.0008 | f1分数: 89.8400 | time: 937.2598
# Step: 13000 | train loss: 0.0046 | f1分数: 89.2091 | time: 940.8865
# Step: 13050 | train loss: 0.0053 | f1分数: 89.4393 | time: 944.4977
# Step: 13100 | train loss: 0.0324 | f1分数: 88.8657 | time: 948.1401
# Step: 13150 | train loss: 0.0117 | f1分数: 89.5123 | time: 951.7782
# Step: 13200 | train loss: 0.0004 | f1分数: 89.4198 | time: 955.4108
# Step: 13250 | train loss: 0.0008 | f1分数: 89.4144 | time: 959.0321
# Step: 13300 | train loss: 0.0021 | f1分数: 88.6734 | time: 962.6453
# Step: 13350 | train loss: 0.0128 | f1分数: 89.3367 | time: 966.2206
# Step: 13400 | train loss: 0.0001 | f1分数: 89.5621 | time: 969.8312
# Step: 13450 | train loss: 0.0184 | f1分数: 89.3665 | time: 973.4593
# Step: 13500 | train loss: 0.0194 | f1分数: 88.4217 | time: 977.0813
# Step: 13550 | train loss: 0.0027 | f1分数: 87.9261 | time: 980.6838
# Step: 13600 | train loss: 0.0179 | f1分数: 88.1173 | time: 984.3196
# Step: 13650 | train loss: 0.0017 | f1分数: 88.1436 | time: 987.9515
# Step: 13700 | train loss: 0.0249 | f1分数: 88.5190 | time: 991.5598
# Step: 13750 | train loss: 0.0152 | f1分数: 89.3320 | time: 995.1697
# Step: 13800 | train loss: 0.0191 | f1分数: 88.9043 | time: 998.7555
# Step: 13850 | train loss: 0.0111 | f1分数: 88.8645 | time: 1002.3491
# Step: 13900 | train loss: 0.0012 | f1分数: 90.0413 | time: 1005.9612

# Step: 50 | train loss: 2.9377 | f1分数: 75.4185 | time: 4.8333
# Step: 100 | train loss: 0.8473 | f1分数: 75.4185 | time: 8.6347
# Step: 150 | train loss: 0.6408 | f1分数: 76.3667 | time: 12.4045
# Step: 200 | train loss: 0.4137 | f1分数: 77.1917 | time: 16.2377
# Step: 250 | train loss: 0.6366 | f1分数: 78.4521 | time: 20.0540
# Step: 300 | train loss: 1.3577 | f1分数: 79.6313 | time: 23.8751
# Step: 350 | train loss: 0.3473 | f1分数: 81.3949 | time: 27.6869
# Step: 400 | train loss: 0.8032 | f1分数: 81.7358 | time: 31.5191
# Step: 450 | train loss: 0.5152 | f1分数: 83.2469 | time: 35.3597
# Step: 500 | train loss: 0.7198 | f1分数: 83.3385 | time: 39.1686
# Step: 550 | train loss: 0.7176 | f1分数: 80.7021 | time: 43.0336
# Step: 600 | train loss: 0.3332 | f1分数: 82.4086 | time: 46.8677
# Step: 650 | train loss: 0.4052 | f1分数: 82.9364 | time: 50.6781
# Step: 700 | train loss: 0.5578 | f1分数: 84.4231 | time: 54.5280
# Step: 750 | train loss: 0.2898 | f1分数: 84.3354 | time: 58.3342
# Step: 800 | train loss: 0.3776 | f1分数: 82.9132 | time: 62.1331
# Step: 850 | train loss: 0.3903 | f1分数: 83.6421 | time: 65.9881
# Step: 900 | train loss: 0.3930 | f1分数: 85.0514 | time: 69.7966
# Step: 950 | train loss: 0.3865 | f1分数: 86.1440 | time: 73.6076
# Step: 1000 | train loss: 0.3290 | f1分数: 87.1311 | time: 77.4331
# Step: 1050 | train loss: 0.3883 | f1分数: 86.5966 | time: 81.2686
# Step: 1100 | train loss: 0.3833 | f1分数: 86.2482 | time: 85.0681
# Step: 1150 | train loss: 0.5163 | f1分数: 88.7637 | time: 88.8779
# Step: 1200 | train loss: 0.3332 | f1分数: 86.6977 | time: 92.6960
# Step: 1250 | train loss: 0.3800 | f1分数: 85.6342 | time: 96.5145
# Step: 1300 | train loss: 0.4787 | f1分数: 86.4868 | time: 100.3058
# Step: 1350 | train loss: 0.3680 | f1分数: 86.9541 | time: 104.1471
# Step: 1400 | train loss: 0.0584 | f1分数: 85.7912 | time: 107.9602
# Step: 1450 | train loss: 0.2192 | f1分数: 84.4656 | time: 111.7857
# Step: 1500 | train loss: 0.1523 | f1分数: 84.2829 | time: 115.6082
# Step: 1550 | train loss: 0.1354 | f1分数: 83.8935 | time: 119.4093
# Step: 1600 | train loss: 0.2036 | f1分数: 85.5193 | time: 123.2090
# Step: 1650 | train loss: 0.2663 | f1分数: 87.8546 | time: 127.0507
# Step: 1700 | train loss: 0.1807 | f1分数: 88.2382 | time: 130.8846
# Step: 1750 | train loss: 0.1892 | f1分数: 87.8459 | time: 134.6859
# Step: 1800 | train loss: 0.2381 | f1分数: 88.6832 | time: 138.4991
# Step: 1850 | train loss: 0.2056 | f1分数: 89.3763 | time: 142.3027
# Step: 1900 | train loss: 0.1373 | f1分数: 88.7017 | time: 146.1393
# Step: 1950 | train loss: 0.1492 | f1分数: 88.7575 | time: 149.9432
# Step: 2000 | train loss: 0.2479 | f1分数: 87.5553 | time: 153.7462
# Step: 2050 | train loss: 0.1840 | f1分数: 87.9563 | time: 157.5316
# Step: 2100 | train loss: 0.1348 | f1分数: 86.2625 | time: 161.3429
# Step: 2150 | train loss: 0.2036 | f1分数: 84.0361 | time: 165.1434
# Step: 2200 | train loss: 0.1907 | f1分数: 78.8626 | time: 168.9211
# Step: 2250 | train loss: 0.1149 | f1分数: 84.4035 | time: 172.6977
# Step: 2300 | train loss: 0.1956 | f1分数: 88.6793 | time: 176.5156
# Step: 2350 | train loss: 0.0795 | f1分数: 86.7787 | time: 180.3362
# Step: 2400 | train loss: 0.1509 | f1分数: 87.5810 | time: 184.1512
# Step: 2450 | train loss: 0.0818 | f1分数: 88.1788 | time: 187.9474
# Step: 2500 | train loss: 0.1573 | f1分数: 89.9586 | time: 191.7220
# Step: 2550 | train loss: 0.0577 | f1分数: 89.7684 | time: 195.5340
# Step: 2600 | train loss: 0.0758 | f1分数: 89.3393 | time: 199.3565
# Step: 2650 | train loss: 0.2471 | f1分数: 86.8272 | time: 203.1626
# Step: 2700 | train loss: 0.1362 | f1分数: 87.8765 | time: 206.9827
# Step: 2750 | train loss: 0.0567 | f1分数: 84.4565 | time: 210.7909
# Step: 2800 | train loss: 0.0617 | f1分数: 86.0147 | time: 214.5891
# Step: 2850 | train loss: 0.0948 | f1分数: 85.0955 | time: 218.4081
# Step: 2900 | train loss: 0.0909 | f1分数: 87.0545 | time: 222.1834
# Step: 2950 | train loss: 0.1764 | f1分数: 85.0829 | time: 226.0140
# Step: 3000 | train loss: 0.0504 | f1分数: 84.1692 | time: 229.8294
# Step: 3050 | train loss: 0.1194 | f1分数: 89.3281 | time: 233.6115
# Step: 3100 | train loss: 0.0961 | f1分数: 88.4072 | time: 237.4513
# Step: 3150 | train loss: 0.0956 | f1分数: 89.9771 | time: 241.2259
# Step: 3200 | train loss: 0.0985 | f1分数: 89.5971 | time: 245.0438
# Step: 3250 | train loss: 0.0754 | f1分数: 88.8717 | time: 248.8522
# Step: 3300 | train loss: 0.0525 | f1分数: 84.4269 | time: 252.6920
# Step: 3350 | train loss: 0.0964 | f1分数: 89.1359 | time: 256.4836
# Step: 3400 | train loss: 0.0692 | f1分数: 87.6645 | time: 260.3096
# Step: 3450 | train loss: 0.0984 | f1分数: 88.7757 | time: 264.1100
# Step: 3500 | train loss: 0.0642 | f1分数: 88.0251 | time: 267.8815
# Step: 3550 | train loss: 0.1129 | f1分数: 88.5662 | time: 271.6851
# Step: 3600 | train loss: 0.0321 | f1分数: 89.6394 | time: 275.5045
# Step: 3650 | train loss: 0.0673 | f1分数: 89.8286 | time: 279.2624
# Step: 3700 | train loss: 0.0260 | f1分数: 87.9800 | time: 283.0977
# Step: 3750 | train loss: 0.0217 | f1分数: 89.3665 | time: 286.9173
# Step: 3800 | train loss: 0.1298 | f1分数: 89.2988 | time: 290.6742
# Step: 3850 | train loss: 0.0582 | f1分数: 89.4848 | time: 294.5348
# Step: 3900 | train loss: 0.0570 | f1分数: 89.1118 | time: 298.3740
# Step: 3950 | train loss: 0.0425 | f1分数: 88.8814 | time: 302.1895
# Step: 4000 | train loss: 0.1082 | f1分数: 90.2899 | time: 305.9975
# Step: 4050 | train loss: 0.0211 | f1分数: 88.2626 | time: 309.8444
# Step: 4100 | train loss: 0.0308 | f1分数: 88.5443 | time: 313.6399
# Step: 4150 | train loss: 0.0268 | f1分数: 89.3728 | time: 317.4641
# Step: 4200 | train loss: 0.0214 | f1分数: 88.4399 | time: 321.2607
# Step: 4250 | train loss: 0.0231 | f1分数: 89.0587 | time: 325.0716
# Step: 4300 | train loss: 0.0363 | f1分数: 89.7888 | time: 328.9017
# Step: 4350 | train loss: 0.0251 | f1分数: 89.3662 | time: 332.6955
# Step: 4400 | train loss: 0.0208 | f1分数: 88.5832 | time: 336.5072
# Step: 4450 | train loss: 0.0104 | f1分数: 88.6976 | time: 340.3144
# Step: 4500 | train loss: 0.0437 | f1分数: 89.9814 | time: 344.1130
# Step: 4550 | train loss: 0.0147 | f1分数: 89.8124 | time: 347.9632
# Step: 4600 | train loss: 0.0122 | f1分数: 90.2169 | time: 351.7841
# Step: 4650 | train loss: 0.0406 | f1分数: 89.8866 | time: 355.6097
# Step: 4700 | train loss: 0.0592 | f1分数: 88.8990 | time: 359.4054
# Step: 4750 | train loss: 0.0500 | f1分数: 88.7114 | time: 363.2239
# Step: 4800 | train loss: 0.0933 | f1分数: 87.9757 | time: 367.0223
# Step: 4850 | train loss: 0.0202 | f1分数: 89.6601 | time: 370.8272
# Step: 4900 | train loss: 0.0105 | f1分数: 89.0367 | time: 374.6451
# Step: 4950 | train loss: 0.0672 | f1分数: 88.9426 | time: 378.4715
# Step: 5000 | train loss: 0.0248 | f1分数: 89.9411 | time: 382.2720
# Step: 5050 | train loss: 0.0080 | f1分数: 89.4416 | time: 386.0978
# Step: 5100 | train loss: 0.0165 | f1分数: 89.6337 | time: 389.8701
# Step: 5150 | train loss: 0.0660 | f1分数: 89.3031 | time: 393.7034
# Step: 5200 | train loss: 0.0238 | f1分数: 89.2438 | time: 397.5125
# Step: 5250 | train loss: 0.0340 | f1分数: 89.5883 | time: 401.2996
# Step: 5300 | train loss: 0.0689 | f1分数: 89.9471 | time: 405.0920
# Step: 5350 | train loss: 0.0373 | f1分数: 89.2802 | time: 408.8988
# Step: 5400 | train loss: 0.0359 | f1分数: 88.5594 | time: 412.7142
# Step: 5450 | train loss: 0.0629 | f1分数: 89.1301 | time: 416.5640
# Step: 5500 | train loss: 0.0347 | f1分数: 89.0091 | time: 420.3580
# Step: 5550 | train loss: 0.0606 | f1分数: 89.4309 | time: 424.1673
# Step: 5600 | train loss: 0.0182 | f1分数: 89.6386 | time: 427.9706
# Step: 5650 | train loss: 0.0059 | f1分数: 88.5430 | time: 431.7688
# Step: 5700 | train loss: 0.0520 | f1分数: 89.4847 | time: 435.5758
# Step: 5750 | train loss: 0.0129 | f1分数: 89.5448 | time: 439.3896
# Step: 5800 | train loss: 0.0727 | f1分数: 89.8954 | time: 443.1929
# Step: 5850 | train loss: 0.0241 | f1分数: 88.7124 | time: 446.9938
# Step: 5900 | train loss: 0.0277 | f1分数: 89.0525 | time: 450.8282
# Step: 5950 | train loss: 0.0018 | f1分数: 89.1753 | time: 454.6750
# Step: 6000 | train loss: 0.0420 | f1分数: 88.6818 | time: 458.4908
# Step: 6050 | train loss: 0.0464 | f1分数: 88.9176 | time: 462.3166
# Step: 6100 | train loss: 0.0069 | f1分数: 89.2300 | time: 466.1596
# Step: 6150 | train loss: 0.0224 | f1分数: 88.0178 | time: 469.9902
# Step: 6200 | train loss: 0.0162 | f1分数: 89.1074 | time: 473.8035
# Step: 6250 | train loss: 0.0057 | f1分数: 88.9417 | time: 477.6072
# Step: 6300 | train loss: 0.0309 | f1分数: 88.5557 | time: 481.4261
# Step: 6350 | train loss: 0.0118 | f1分数: 88.6425 | time: 485.2589
# Step: 6400 | train loss: 0.0127 | f1分数: 89.4503 | time: 489.0805
# Step: 6450 | train loss: 0.0664 | f1分数: 89.8031 | time: 492.8989
# Step: 6500 | train loss: 0.0226 | f1分数: 88.6760 | time: 496.7198
# Step: 6550 | train loss: 0.0220 | f1分数: 89.7504 | time: 500.5436
# Step: 6600 | train loss: 0.0193 | f1分数: 88.5431 | time: 504.3620
# Step: 6650 | train loss: 0.0430 | f1分数: 89.6176 | time: 508.1635
# Step: 6700 | train loss: 0.0043 | f1分数: 89.2069 | time: 511.9833
# Step: 6750 | train loss: 0.0039 | f1分数: 89.0964 | time: 515.7701
# Step: 6800 | train loss: 0.0317 | f1分数: 88.5460 | time: 519.5465
# Step: 6850 | train loss: 0.0290 | f1分数: 88.6138 | time: 523.3290
# Step: 6900 | train loss: 0.0316 | f1分数: 89.0050 | time: 527.1452
# Step: 6950 | train loss: 0.0158 | f1分数: 88.2579 | time: 530.9380
# Step: 7000 | train loss: 0.0011 | f1分数: 88.9495 | time: 534.7561
# Step: 7050 | train loss: 0.0032 | f1分数: 89.8662 | time: 538.5364
# Step: 7100 | train loss: 0.0048 | f1分数: 88.7216 | time: 542.3593
# Step: 7150 | train loss: 0.0234 | f1分数: 88.9158 | time: 546.1876
# Step: 7200 | train loss: 0.0051 | f1分数: 89.4768 | time: 550.0270
# Step: 7250 | train loss: 0.0099 | f1分数: 89.4272 | time: 553.8166
# Step: 7300 | train loss: 0.0232 | f1分数: 85.2681 | time: 557.6148
# Step: 7350 | train loss: 0.0086 | f1分数: 89.5375 | time: 561.4516
# Step: 7400 | train loss: 0.0341 | f1分数: 88.1153 | time: 565.2532
# Step: 7450 | train loss: 0.0817 | f1分数: 87.5750 | time: 569.0900
# Step: 7500 | train loss: 0.0021 | f1分数: 89.0283 | time: 572.9000
# Step: 7550 | train loss: 0.0062 | f1分数: 88.6991 | time: 576.7184
# Step: 7600 | train loss: 0.0157 | f1分数: 89.2659 | time: 580.5248
# Step: 7650 | train loss: 0.0053 | f1分数: 89.2551 | time: 584.3080
# Step: 7700 | train loss: 0.0028 | f1分数: 89.7265 | time: 588.1196
# Step: 7750 | train loss: 0.0221 | f1分数: 89.5342 | time: 591.9338
# Step: 7800 | train loss: 0.0264 | f1分数: 89.8570 | time: 595.7658
# Step: 7850 | train loss: 0.0088 | f1分数: 89.4401 | time: 599.5971
# Step: 7900 | train loss: 0.0278 | f1分数: 89.3158 | time: 603.4344
# Step: 7950 | train loss: 0.0091 | f1分数: 89.8828 | time: 607.2444
# Step: 8000 | train loss: 0.0140 | f1分数: 89.3436 | time: 611.0716
# Step: 8050 | train loss: 0.0169 | f1分数: 89.0226 | time: 614.8974
# Step: 8100 | train loss: 0.0069 | f1分数: 89.0863 | time: 618.6661
# Step: 8150 | train loss: 0.0170 | f1分数: 89.7346 | time: 622.4551
# Step: 8200 | train loss: 0.0317 | f1分数: 89.5065 | time: 626.2599
# Step: 8250 | train loss: 0.0904 | f1分数: 89.2642 | time: 630.0786
# Step: 8300 | train loss: 0.0025 | f1分数: 88.8427 | time: 633.9293
# Step: 8350 | train loss: 0.0342 | f1分数: 89.1776 | time: 637.7593
# Step: 8400 | train loss: 0.0069 | f1分数: 89.5958 | time: 641.5823
# Step: 8450 | train loss: 0.0366 | f1分数: 89.9197 | time: 645.3751
# Step: 8500 | train loss: 0.0154 | f1分数: 89.0302 | time: 649.1360
# Step: 8550 | train loss: 0.0154 | f1分数: 89.2382 | time: 652.9927
# Step: 8600 | train loss: 0.0073 | f1分数: 89.4124 | time: 656.8314
# Step: 8650 | train loss: 0.0244 | f1分数: 87.4292 | time: 660.7106
# Step: 8700 | train loss: 0.0000 | f1分数: 87.9869 | time: 664.5310
# Step: 8750 | train loss: 0.0008 | f1分数: 88.9103 | time: 668.3572
# Step: 8800 | train loss: 0.0051 | f1分数: 89.6386 | time: 672.1828
# Step: 8850 | train loss: 0.0062 | f1分数: 89.3374 | time: 676.0358
# Step: 8900 | train loss: 0.0008 | f1分数: 88.7949 | time: 679.8500
# Step: 8950 | train loss: 0.0528 | f1分数: 87.7895 | time: 683.6486
# Step: 9000 | train loss: 0.0016 | f1分数: 88.8051 | time: 687.4805
# Step: 9050 | train loss: 0.0022 | f1分数: 89.9778 | time: 691.3211
# Step: 9100 | train loss: 0.0081 | f1分数: 89.6475 | time: 695.0970
# Step: 9150 | train loss: 0.0091 | f1分数: 90.2392 | time: 698.9346
# Step: 9200 | train loss: 0.0015 | f1分数: 89.1298 | time: 702.7338
# Step: 9250 | train loss: 0.0101 | f1分数: 90.0956 | time: 706.5373
# Step: 9300 | train loss: 0.0101 | f1分数: 89.4558 | time: 710.3522
# Step: 9350 | train loss: 0.0059 | f1分数: 89.6957 | time: 714.1686
# Step: 9400 | train loss: 0.0056 | f1分数: 88.1817 | time: 717.9477
# Step: 9450 | train loss: 0.0249 | f1分数: 88.5455 | time: 721.7522
# Step: 9500 | train loss: 0.0021 | f1分数: 88.7940 | time: 725.5603
# Step: 9550 | train loss: 0.0048 | f1分数: 89.6639 | time: 729.3415
# Step: 9600 | train loss: 0.0230 | f1分数: 89.8863 | time: 733.1395
# Step: 9650 | train loss: 0.0074 | f1分数: 89.7482 | time: 736.9244
# Step: 9700 | train loss: 0.0110 | f1分数: 89.3884 | time: 740.7115
# Step: 9750 | train loss: 0.0018 | f1分数: 90.0398 | time: 744.5387
# Step: 9800 | train loss: 0.0251 | f1分数: 90.1485 | time: 748.3352
# Step: 9850 | train loss: 0.0015 | f1分数: 89.9644 | time: 752.1320
# Step: 9900 | train loss: 0.0033 | f1分数: 90.4521 | time: 755.9555
# Step: 9950 | train loss: 0.0074 | f1分数: 90.0111 | time: 759.7527
# Step: 10000 | train loss: 0.0519 | f1分数: 89.6871 | time: 763.5707
# Step: 10050 | train loss: 0.0218 | f1分数: 89.2264 | time: 767.3957
# Step: 10100 | train loss: 0.0147 | f1分数: 89.3991 | time: 771.1907
# Step: 10150 | train loss: 0.0038 | f1分数: 89.1144 | time: 774.9860
# Step: 10200 | train loss: 0.0047 | f1分数: 88.3882 | time: 778.7873
# Step: 10250 | train loss: 0.0158 | f1分数: 90.1119 | time: 782.5570
# Step: 10300 | train loss: 0.0019 | f1分数: 89.6131 | time: 786.3864
# Step: 10350 | train loss: 0.0227 | f1分数: 89.5626 | time: 790.1948
# Step: 10400 | train loss: 0.0040 | f1分数: 90.3829 | time: 794.0206
# Step: 10450 | train loss: 0.0075 | f1分数: 90.4949 | time: 797.8255
# Step: 10500 | train loss: 0.0036 | f1分数: 90.1642 | time: 801.6244
# Step: 10550 | train loss: 0.0001 | f1分数: 90.3318 | time: 805.4287
# Step: 10600 | train loss: 0.0001 | f1分数: 89.4717 | time: 809.2277
# Step: 10650 | train loss: 0.0091 | f1分数: 90.7976 | time: 813.0142
# Step: 10700 | train loss: 0.0019 | f1分数: 90.0676 | time: 816.7962
# Step: 10750 | train loss: 0.0015 | f1分数: 87.1415 | time: 820.6288
# Step: 10800 | train loss: 0.0149 | f1分数: 87.1454 | time: 824.4309
# Step: 10850 | train loss: 0.0159 | f1分数: 90.5162 | time: 828.2194
# Step: 10900 | train loss: 0.0029 | f1分数: 89.8804 | time: 832.0254
# Step: 10950 | train loss: 0.0015 | f1分数: 89.4255 | time: 835.8145
# Step: 11000 | train loss: 0.0085 | f1分数: 89.2366 | time: 839.6504
# Step: 11050 | train loss: 0.0111 | f1分数: 89.2028 | time: 843.4346
# Step: 11100 | train loss: 0.0452 | f1分数: 89.5150 | time: 847.2193
# Step: 11150 | train loss: 0.0070 | f1分数: 90.0843 | time: 851.0262
# Step: 11200 | train loss: 0.0372 | f1分数: 90.1636 | time: 854.8267
# Step: 11250 | train loss: 0.0078 | f1分数: 89.0921 | time: 858.6598
# Step: 11300 | train loss: 0.0343 | f1分数: 90.0334 | time: 862.4437
# Step: 11350 | train loss: 0.0014 | f1分数: 89.8816 | time: 866.1980
# Step: 11400 | train loss: 0.0024 | f1分数: 89.9306 | time: 869.9672
# Step: 11450 | train loss: 0.0019 | f1分数: 89.7271 | time: 873.7995
# Step: 11500 | train loss: 0.0045 | f1分数: 90.2089 | time: 877.6104
# Step: 11550 | train loss: 0.0043 | f1分数: 89.4709 | time: 881.4109
# Step: 11600 | train loss: 0.0025 | f1分数: 90.2143 | time: 885.2255
# Step: 11650 | train loss: 0.0003 | f1分数: 90.1607 | time: 889.0548
# Step: 11700 | train loss: 0.0034 | f1分数: 89.9748 | time: 892.8493
# Step: 11750 | train loss: 0.0096 | f1分数: 90.0221 | time: 896.6607
# Step: 11800 | train loss: 0.0124 | f1分数: 89.7246 | time: 900.5088
# Step: 11850 | train loss: 0.0010 | f1分数: 90.0016 | time: 904.3168
# Step: 11900 | train loss: 0.0018 | f1分数: 89.9538 | time: 908.1120
# Step: 11950 | train loss: 0.0010 | f1分数: 90.2191 | time: 911.9032
# Step: 12000 | train loss: 0.0030 | f1分数: 88.7770 | time: 915.7153
# Step: 12050 | train loss: 0.0325 | f1分数: 90.4246 | time: 919.5260
# Step: 12100 | train loss: 0.0301 | f1分数: 90.4521 | time: 923.3654
# Step: 12150 | train loss: 0.0015 | f1分数: 84.3889 | time: 927.1568
# Step: 12200 | train loss: 0.0463 | f1分数: 84.5366 | time: 930.9762
# Step: 12250 | train loss: 0.0336 | f1分数: 86.2620 | time: 934.7452
# Step: 12300 | train loss: 0.0058 | f1分数: 89.3198 | time: 938.5275
# Step: 12350 | train loss: 0.0260 | f1分数: 88.3817 | time: 942.2381
# Step: 12400 | train loss: 0.0096 | f1分数: 89.1439 | time: 945.9167
# Step: 12450 | train loss: 0.0119 | f1分数: 90.2037 | time: 949.5724
# Step: 12500 | train loss: 0.0030 | f1分数: 89.5962 | time: 953.2377
# Step: 12550 | train loss: 0.0003 | f1分数: 89.9222 | time: 956.8713
# Step: 12600 | train loss: 0.0100 | f1分数: 90.3605 | time: 960.5498
# Step: 12650 | train loss: 0.0128 | f1分数: 89.3725 | time: 964.1989
# Step: 12700 | train loss: 0.0096 | f1分数: 89.8750 | time: 967.8264
# Step: 12750 | train loss: 0.0045 | f1分数: 89.9887 | time: 971.5020
# Step: 12800 | train loss: 0.0030 | f1分数: 89.9779 | time: 975.1797
# Step: 12850 | train loss: 0.0022 | f1分数: 89.9832 | time: 978.8234
# Step: 12900 | train loss: 0.0104 | f1分数: 89.4618 | time: 982.4865
# Step: 12950 | train loss: 0.0003 | f1分数: 89.7005 | time: 986.1898
# Step: 13000 | train loss: 0.0312 | f1分数: 90.3149 | time: 989.8717
# Step: 13050 | train loss: 0.0028 | f1分数: 89.8224 | time: 993.5837
# Step: 13100 | train loss: 0.0305 | f1分数: 90.1565 | time: 997.2832
# Step: 13150 | train loss: 0.0081 | f1分数: 90.2916 | time: 1000.9612
# Step: 13200 | train loss: 0.0006 | f1分数: 89.4281 | time: 1004.6472
# Step: 13250 | train loss: 0.0002 | f1分数: 90.4506 | time: 1008.3458
# Step: 13300 | train loss: 0.0328 | f1分数: 90.0493 | time: 1011.9985
# Step: 13350 | train loss: 0.0018 | f1分数: 89.0859 | time: 1015.6631
# Step: 13400 | train loss: 0.0044 | f1分数: 90.1137 | time: 1019.3267
# Step: 13450 | train loss: 0.0080 | f1分数: 90.3685 | time: 1023.0037
# Step: 13500 | train loss: 0.0046 | f1分数: 90.1274 | time: 1026.6961
# Step: 13550 | train loss: 0.0000 | f1分数: 90.2418 | time: 1030.3749
# Step: 13600 | train loss: 0.0000 | f1分数: 90.5228 | time: 1034.0911
# Step: 13650 | train loss: 0.0001 | f1分数: 89.9983 | time: 1037.7609
# Step: 13700 | train loss: 0.0002 | f1分数: 90.1225 | time: 1041.4204
# Step: 13750 | train loss: 0.0060 | f1分数: 90.0724 | time: 1045.0867
# Step: 13800 | train loss: 0.0040 | f1分数: 89.4569 | time: 1048.7905
# Step: 13850 | train loss: 0.0022 | f1分数: 89.9833 | time: 1052.4854
# Step: 13900 | train loss: 0.0007 | f1分数: 89.2860 | time: 1056.2616

# batch_norm 有时不收敛
# Step: 50 | train loss: 6.1624 | f1分数: 75.6440 | time: 6.8038
# Step: 100 | train loss: 0.6827 | f1分数: 72.0924 | time: 12.2992
# Step: 150 | train loss: 0.7133 | f1分数: 79.9552 | time: 17.8217
# Step: 200 | train loss: 0.5860 | f1分数: 79.5907 | time: 23.2922
# Step: 250 | train loss: 0.6444 | f1分数: 78.3384 | time: 28.8088
# Step: 300 | train loss: 1.0326 | f1分数: 77.9670 | time: 34.3258
# Step: 350 | train loss: 0.2327 | f1分数: 79.0226 | time: 39.8617
# Step: 400 | train loss: 0.6817 | f1分数: 82.2296 | time: 45.3612
# Step: 450 | train loss: 0.4483 | f1分数: 83.8589 | time: 50.8759
# Step: 500 | train loss: 0.8613 | f1分数: 85.0670 | time: 56.3891
# Step: 550 | train loss: 0.5844 | f1分数: 80.3591 | time: 61.8905
# Step: 600 | train loss: 0.3514 | f1分数: 79.8306 | time: 67.4193
# Step: 650 | train loss: 0.3183 | f1分数: 82.9498 | time: 72.9548
# Step: 700 | train loss: 0.5136 | f1分数: 85.4735 | time: 78.5216
# Step: 750 | train loss: 0.2157 | f1分数: 86.3608 | time: 84.0529
# Step: 800 | train loss: 0.2817 | f1分数: 85.3448 | time: 89.5876
# Step: 850 | train loss: 0.3543 | f1分数: 85.4754 | time: 95.1126
# Step: 900 | train loss: 0.4859 | f1分数: 86.2118 | time: 100.6356
# Step: 950 | train loss: 0.3596 | f1分数: 84.9699 | time: 106.1597
# Step: 1000 | train loss: 0.2437 | f1分数: 87.3079 | time: 111.6700
# Step: 1050 | train loss: 0.4175 | f1分数: 84.2389 | time: 117.1908
# Step: 1100 | train loss: 0.3431 | f1分数: 86.2320 | time: 122.7177
# Step: 1150 | train loss: 0.4579 | f1分数: 87.4982 | time: 128.2654
# Step: 1200 | train loss: 0.3364 | f1分数: 85.8631 | time: 133.7862
# Step: 1250 | train loss: 0.3977 | f1分数: 83.7694 | time: 139.3370
# Step: 1300 | train loss: 0.3270 | f1分数: 87.7379 | time: 144.9005
# Step: 1350 | train loss: 0.3841 | f1分数: 86.7484 | time: 150.4351
# Step: 1400 | train loss: 0.0936 | f1分数: 85.4956 | time: 155.9378
# Step: 1450 | train loss: 0.1320 | f1分数: 84.2340 | time: 161.4839
# Step: 1500 | train loss: 0.1227 | f1分数: 81.5396 | time: 167.0105
# Step: 1550 | train loss: 0.1120 | f1分数: 85.1921 | time: 172.5547
# Step: 1600 | train loss: 0.1745 | f1分数: 84.9287 | time: 178.0631
# Step: 1650 | train loss: 0.1533 | f1分数: 85.0647 | time: 183.6081
# Step: 1700 | train loss: 0.2199 | f1分数: 85.5567 | time: 189.1561
# Step: 1750 | train loss: 0.1797 | f1分数: 88.1841 | time: 194.6903
# Step: 1800 | train loss: 0.1988 | f1分数: 87.6353 | time: 200.2453
# Step: 1850 | train loss: 0.2368 | f1分数: 88.6638 | time: 205.7805
# Step: 1900 | train loss: 0.1320 | f1分数: 86.2488 | time: 211.3018
# Step: 1950 | train loss: 0.1581 | f1分数: 88.3832 | time: 216.8239
# Step: 2000 | train loss: 0.1943 | f1分数: 87.9442 | time: 222.2958
# Step: 2050 | train loss: 0.1999 | f1分数: 85.0470 | time: 227.7899
# Step: 2100 | train loss: 0.1026 | f1分数: 85.6853 | time: 233.3045
# Step: 2150 | train loss: 0.1789 | f1分数: 86.4110 | time: 238.8367
# Step: 2200 | train loss: 0.0977 | f1分数: 84.9682 | time: 244.3650
# Step: 2250 | train loss: 0.0921 | f1分数: 86.6251 | time: 249.8806
# Step: 2300 | train loss: 0.1603 | f1分数: 88.6037 | time: 255.4084
# Step: 2350 | train loss: 0.0809 | f1分数: 88.2004 | time: 260.9218
# Step: 2400 | train loss: 0.1420 | f1分数: 87.8888 | time: 266.4626
# Step: 2450 | train loss: 0.0307 | f1分数: 88.0847 | time: 271.9810
# Step: 2500 | train loss: 0.1256 | f1分数: 88.8120 | time: 277.5315
# Step: 2550 | train loss: 0.0651 | f1分数: 81.5687 | time: 283.0852
# Step: 2600 | train loss: 0.0469 | f1分数: 89.6185 | time: 288.6289
# Step: 2650 | train loss: 0.0814 | f1分数: 89.4767 | time: 294.1436
# Step: 2700 | train loss: 0.1105 | f1分数: 89.1720 | time: 299.6827
# Step: 2750 | train loss: 0.0316 | f1分数: 84.8975 | time: 305.1966
# Step: 2800 | train loss: 0.0715 | f1分数: 85.5939 | time: 310.7348
# Step: 2850 | train loss: 0.0587 | f1分数: 88.4033 | time: 316.2502
# Step: 2900 | train loss: 0.0538 | f1分数: 85.4697 | time: 321.7948
# Step: 2950 | train loss: 0.1703 | f1分数: 83.9688 | time: 327.3220
# Step: 3000 | train loss: 0.0394 | f1分数: 85.6056 | time: 332.8227
# Step: 3050 | train loss: 0.2071 | f1分数: 86.1684 | time: 338.3156
# Step: 3100 | train loss: 0.0928 | f1分数: 88.4666 | time: 343.8239
# Step: 3150 | train loss: 0.0991 | f1分数: 89.3331 | time: 349.3580
# Step: 3200 | train loss: 0.1110 | f1分数: 87.9341 | time: 354.9110
# Step: 3250 | train loss: 0.0455 | f1分数: 86.9062 | time: 360.4854
# Step: 3300 | train loss: 0.0276 | f1分数: 88.3531 | time: 366.0768
# Step: 3350 | train loss: 0.0764 | f1分数: 88.7156 | time: 371.6079
# Step: 3400 | train loss: 0.0293 | f1分数: 89.4804 | time: 377.0963
# Step: 3450 | train loss: 0.0356 | f1分数: 88.6048 | time: 382.6203
# Step: 3500 | train loss: 0.0408 | f1分数: 88.0333 | time: 388.1636
# Step: 3550 | train loss: 0.1114 | f1分数: 89.3862 | time: 393.6828
# Step: 3600 | train loss: 0.0417 | f1分数: 89.7095 | time: 399.2063
# Step: 3650 | train loss: 0.0343 | f1分数: 89.5731 | time: 404.7596
# Step: 3700 | train loss: 0.0242 | f1分数: 88.8906 | time: 410.2914
# Step: 3750 | train loss: 0.0332 | f1分数: 89.3784 | time: 415.8546
# Step: 3800 | train loss: 0.0939 | f1分数: 89.1235 | time: 421.3966
# Step: 3850 | train loss: 0.0532 | f1分数: 89.8395 | time: 426.9261
# Step: 3900 | train loss: 0.0471 | f1分数: 89.5004 | time: 432.4585
# Step: 3950 | train loss: 0.0887 | f1分数: 85.8414 | time: 438.0032
# Step: 4000 | train loss: 0.0665 | f1分数: 89.5515 | time: 443.5321
# Step: 4050 | train loss: 0.0138 | f1分数: 89.3689 | time: 449.0487
# Step: 4100 | train loss: 0.0576 | f1分数: 89.3515 | time: 454.6046
# Step: 4150 | train loss: 0.0428 | f1分数: 89.2064 | time: 460.1680
# Step: 4200 | train loss: 0.0211 | f1分数: 89.3412 | time: 465.7099
# Step: 4250 | train loss: 0.0239 | f1分数: 88.9543 | time: 471.2960
# Step: 4300 | train loss: 0.0554 | f1分数: 89.4290 | time: 476.8186
# Step: 4350 | train loss: 0.0087 | f1分数: 87.7790 | time: 482.3683
# Step: 4400 | train loss: 0.1530 | f1分数: 88.7346 | time: 487.9010
# Step: 4450 | train loss: 0.0100 | f1分数: 89.0616 | time: 493.4207
# Step: 4500 | train loss: 0.0793 | f1分数: 81.2124 | time: 498.9680
# Step: 4550 | train loss: 0.0512 | f1分数: 88.3071 | time: 504.5017
# Step: 4600 | train loss: 0.0727 | f1分数: 88.5155 | time: 510.0417
# Step: 4650 | train loss: 0.0979 | f1分数: 84.7101 | time: 515.5573
# Step: 4700 | train loss: 0.0289 | f1分数: 87.9112 | time: 521.0549
# Step: 4750 | train loss: 0.0426 | f1分数: 89.0071 | time: 526.6085
# Step: 4800 | train loss: 0.0726 | f1分数: 88.2603 | time: 532.1348
# Step: 4850 | train loss: 0.0161 | f1分数: 88.6880 | time: 537.6491
# Step: 4900 | train loss: 0.0094 | f1分数: 88.7757 | time: 543.1492
# Step: 4950 | train loss: 0.0598 | f1分数: 89.1215 | time: 548.6254
# Step: 5000 | train loss: 0.0084 | f1分数: 90.1550 | time: 554.1717
# Step: 5050 | train loss: 0.0066 | f1分数: 88.4780 | time: 559.7419
# Step: 5100 | train loss: 0.0110 | f1分数: 89.5077 | time: 565.2697
# Step: 5150 | train loss: 0.0304 | f1分数: 90.3328 | time: 570.7550
# Step: 5200 | train loss: 0.0145 | f1分数: 90.4397 | time: 576.2771
# Step: 5250 | train loss: 0.0094 | f1分数: 86.8068 | time: 581.8010
# Step: 5300 | train loss: 0.0088 | f1分数: 89.4811 | time: 587.3316
# Step: 5350 | train loss: 0.0390 | f1分数: 89.8052 | time: 592.8577
# Step: 5400 | train loss: 0.0485 | f1分数: 88.9599 | time: 598.3765
# Step: 5450 | train loss: 0.0107 | f1分数: 89.6301 | time: 603.9454
# Step: 5500 | train loss: 0.0169 | f1分数: 89.9459 | time: 609.4656
# Step: 5550 | train loss: 0.0458 | f1分数: 89.6404 | time: 614.9587
# Step: 5600 | train loss: 0.0246 | f1分数: 90.1005 | time: 620.4674
# Step: 5650 | train loss: 0.0038 | f1分数: 89.9348 | time: 625.9808
# Step: 5700 | train loss: 0.0188 | f1分数: 88.9252 | time: 631.5069
# Step: 5750 | train loss: 0.0038 | f1分数: 89.3587 | time: 637.0397
# Step: 5800 | train loss: 0.0324 | f1分数: 89.0503 | time: 642.5623
# Step: 5850 | train loss: 0.0262 | f1分数: 86.7671 | time: 648.0887
# Step: 5900 | train loss: 0.0089 | f1分数: 88.6291 | time: 653.6145
# Step: 5950 | train loss: 0.0007 | f1分数: 89.3368 | time: 659.1236
# Step: 6000 | train loss: 0.0049 | f1分数: 89.8180 | time: 664.6268
# Step: 6050 | train loss: 0.0313 | f1分数: 87.3732 | time: 670.1746
# Step: 6100 | train loss: 0.0218 | f1分数: 86.3730 | time: 675.7108
# Step: 6150 | train loss: 0.0197 | f1分数: 88.5125 | time: 681.2352
# Step: 6200 | train loss: 0.0118 | f1分数: 84.3944 | time: 686.7937
# Step: 6250 | train loss: 0.0117 | f1分数: 86.8003 | time: 692.3361
# Step: 6300 | train loss: 0.0337 | f1分数: 84.6777 | time: 697.8611
# Step: 6350 | train loss: 0.0087 | f1分数: 87.3081 | time: 703.3554
# Step: 6400 | train loss: 0.0116 | f1分数: 89.2378 | time: 708.9378
# Step: 6450 | train loss: 0.1332 | f1分数: 89.5300 | time: 714.4364
# Step: 6500 | train loss: 0.0419 | f1分数: 84.8666 | time: 719.9194
# Step: 6550 | train loss: 0.0123 | f1分数: 89.2523 | time: 725.4296
# Step: 6600 | train loss: 0.0730 | f1分数: 87.9559 | time: 730.9374
# Step: 6650 | train loss: 0.0223 | f1分数: 89.4390 | time: 736.4977
# Step: 6700 | train loss: 0.0131 | f1分数: 90.0714 | time: 741.9610
# Step: 6750 | train loss: 0.0027 | f1分数: 88.2573 | time: 747.4518
# Step: 6800 | train loss: 0.0238 | f1分数: 89.0943 | time: 752.9626
# Step: 6850 | train loss: 0.0031 | f1分数: 89.2729 | time: 758.5033
# Step: 6900 | train loss: 0.0063 | f1分数: 89.5418 | time: 764.0471
# Step: 6950 | train loss: 0.0200 | f1分数: 88.9760 | time: 769.5736
# Step: 7000 | train loss: 0.0042 | f1分数: 88.5391 | time: 775.1241
# Step: 7050 | train loss: 0.0157 | f1分数: 88.5496 | time: 780.6505
# Step: 7100 | train loss: 0.0006 | f1分数: 89.3010 | time: 786.1699
# Step: 7150 | train loss: 0.0370 | f1分数: 89.9451 | time: 791.6750
# Step: 7200 | train loss: 0.0044 | f1分数: 89.1869 | time: 797.2040
# Step: 7250 | train loss: 0.0020 | f1分数: 89.7886 | time: 802.7209
# Step: 7300 | train loss: 0.0911 | f1分数: 83.9841 | time: 808.2795
# Step: 7350 | train loss: 0.0152 | f1分数: 88.7808 | time: 813.8592
# Step: 7400 | train loss: 0.0157 | f1分数: 86.1523 | time: 819.3878
# Step: 7450 | train loss: 0.0152 | f1分数: 87.5490 | time: 824.9463
# Step: 7500 | train loss: 0.0033 | f1分数: 88.1960 | time: 830.4605
# Step: 7550 | train loss: 0.0024 | f1分数: 88.8412 | time: 836.0055
# Step: 7600 | train loss: 0.0268 | f1分数: 88.9333 | time: 841.5465
# Step: 7650 | train loss: 0.0030 | f1分数: 88.9444 | time: 847.0820
# Step: 7700 | train loss: 0.0007 | f1分数: 89.6755 | time: 852.6132
# Step: 7750 | train loss: 0.0325 | f1分数: 88.3132 | time: 858.1337
# Step: 7800 | train loss: 0.0200 | f1分数: 89.8431 | time: 863.6493
# Step: 7850 | train loss: 0.0126 | f1分数: 89.3425 | time: 869.2006
# Step: 7900 | train loss: 0.0080 | f1分数: 88.2341 | time: 874.7664
# Step: 7950 | train loss: 0.0167 | f1分数: 89.0580 | time: 880.3187
# Step: 8000 | train loss: 0.0272 | f1分数: 89.2602 | time: 885.8220
# Step: 8050 | train loss: 0.0492 | f1分数: 86.3493 | time: 891.3875
# Step: 8100 | train loss: 0.0049 | f1分数: 84.6855 | time: 896.9580
# Step: 8150 | train loss: 0.0227 | f1分数: 89.1915 | time: 902.4794
# Step: 8200 | train loss: 0.0377 | f1分数: 88.6853 | time: 908.0048
# Step: 8250 | train loss: 0.0228 | f1分数: 89.5509 | time: 913.5372
# Step: 8300 | train loss: 0.0099 | f1分数: 89.7330 | time: 919.1269
# Step: 8350 | train loss: 0.0508 | f1分数: 89.7576 | time: 924.6399
# Step: 8400 | train loss: 0.0010 | f1分数: 89.8025 | time: 930.1917
# Step: 8450 | train loss: 0.0192 | f1分数: 89.4114 | time: 935.7583
# Step: 8500 | train loss: 0.0123 | f1分数: 90.1354 | time: 941.3053
# Step: 8550 | train loss: 0.0377 | f1分数: 89.3715 | time: 946.8598
# Step: 8600 | train loss: 0.0002 | f1分数: 89.8422 | time: 952.3801
# Step: 8650 | train loss: 0.0031 | f1分数: 90.5818 | time: 957.9277
# Step: 8700 | train loss: 0.0000 | f1分数: 82.6297 | time: 963.5006
# Step: 8750 | train loss: 0.0085 | f1分数: 89.8342 | time: 969.0534
# Step: 8800 | train loss: 0.0019 | f1分数: 89.5506 | time: 974.6420
# Step: 8850 | train loss: 0.0113 | f1分数: 89.0347 | time: 980.1898
# Step: 8900 | train loss: 0.0119 | f1分数: 88.8191 | time: 985.7313
# Step: 8950 | train loss: 0.0110 | f1分数: 90.3117 | time: 991.2618
# Step: 9000 | train loss: 0.0011 | f1分数: 89.5503 | time: 996.8000
# Step: 9050 | train loss: 0.0060 | f1分数: 90.4093 | time: 1002.3780
# Step: 9100 | train loss: 0.0104 | f1分数: 89.7524 | time: 1007.9175
# Step: 9150 | train loss: 0.0019 | f1分数: 88.8510 | time: 1013.4395
# Step: 9200 | train loss: 0.0005 | f1分数: 89.9056 | time: 1018.9368
# Step: 9250 | train loss: 0.0027 | f1分数: 89.1254 | time: 1024.4795
# Step: 9300 | train loss: 0.0149 | f1分数: 89.9721 | time: 1030.0094
# Step: 9350 | train loss: 0.0014 | f1分数: 88.8723 | time: 1035.5543
# Step: 9400 | train loss: 0.0001 | f1分数: 89.9771 | time: 1041.0663
# Step: 9450 | train loss: 0.0191 | f1分数: 90.0147 | time: 1046.5762
# Step: 9500 | train loss: 0.0011 | f1分数: 90.0646 | time: 1052.1645
# Step: 9550 | train loss: 0.0032 | f1分数: 89.7891 | time: 1057.6806
# Step: 9600 | train loss: 0.0281 | f1分数: 89.6508 | time: 1063.2072
# Step: 9650 | train loss: 0.0196 | f1分数: 88.9977 | time: 1068.7468
# Step: 9700 | train loss: 0.0175 | f1分数: 89.4261 | time: 1074.2436
# Step: 9750 | train loss: 0.0002 | f1分数: 89.4659 | time: 1079.7757
# Step: 9800 | train loss: 0.0366 | f1分数: 88.6198 | time: 1085.3195
# Step: 9850 | train loss: 0.0126 | f1分数: 88.3566 | time: 1090.8730
# Step: 9900 | train loss: 0.0057 | f1分数: 88.1551 | time: 1096.4040
# Step: 9950 | train loss: 0.0120 | f1分数: 88.6406 | time: 1101.9758
# Step: 10000 | train loss: 0.0218 | f1分数: 88.3875 | time: 1107.5316
# Step: 10050 | train loss: 0.0079 | f1分数: 87.2006 | time: 1113.0414
# Step: 10100 | train loss: 0.0207 | f1分数: 88.2155 | time: 1118.5532
# Step: 10150 | train loss: 0.0155 | f1分数: 88.7937 | time: 1124.0563
# Step: 10200 | train loss: 0.0085 | f1分数: 88.3784 | time: 1129.6296
# Step: 10250 | train loss: 0.0211 | f1分数: 89.6176 | time: 1135.1635
# Step: 10300 | train loss: 0.0031 | f1分数: 89.6082 | time: 1140.6596
# Step: 10350 | train loss: 0.0060 | f1分数: 90.0622 | time: 1146.1701
# Step: 10400 | train loss: 0.0139 | f1分数: 88.5532 | time: 1151.6908
# Step: 10450 | train loss: 0.0134 | f1分数: 90.1449 | time: 1157.2082
# Step: 10500 | train loss: 0.0014 | f1分数: 90.2483 | time: 1162.7605
# Step: 10550 | train loss: 0.0004 | f1分数: 89.3756 | time: 1168.3292
# Step: 10600 | train loss: 0.0007 | f1分数: 90.0208 | time: 1173.8811
# Step: 10650 | train loss: 0.0001 | f1分数: 89.7611 | time: 1179.4163
# Step: 10700 | train loss: 0.0030 | f1分数: 90.0163 | time: 1184.9720
# Step: 10750 | train loss: 0.0001 | f1分数: 90.0170 | time: 1190.5212
# Step: 10800 | train loss: 0.0106 | f1分数: 89.9905 | time: 1196.0076
# Step: 10850 | train loss: 0.0035 | f1分数: 89.9887 | time: 1201.5538
# Step: 10900 | train loss: 0.0001 | f1分数: 89.9843 | time: 1207.1230
# Step: 10950 | train loss: 0.0001 | f1分数: 90.3111 | time: 1212.6352
# Step: 11000 | train loss: 0.0031 | f1分数: 89.9251 | time: 1218.1977
# Step: 11050 | train loss: 0.0001 | f1分数: 90.1379 | time: 1223.7375
# Step: 11100 | train loss: 0.0001 | f1分数: 90.1303 | time: 1229.2907
# Step: 11150 | train loss: 0.0001 | f1分数: 90.2264 | time: 1234.8607
# Step: 11200 | train loss: 0.0004 | f1分数: 90.2082 | time: 1240.4127
# Step: 11250 | train loss: 0.0000 | f1分数: 90.2048 | time: 1245.9922
# Step: 11300 | train loss: 0.0000 | f1分数: 90.0571 | time: 1251.5408
# Step: 11350 | train loss: 0.0000 | f1分数: 90.1689 | time: 1257.0580
# Step: 11400 | train loss: 0.0000 | f1分数: 89.7236 | time: 1262.5709
# Step: 11450 | train loss: 0.0000 | f1分数: 90.1170 | time: 1268.1169
# Step: 11500 | train loss: 0.0011 | f1分数: 90.0157 | time: 1273.7035
# Step: 11550 | train loss: 0.0000 | f1分数: 90.1941 | time: 1279.2358
# Step: 11600 | train loss: 0.0000 | f1分数: 89.9016 | time: 1284.8331
# Step: 11650 | train loss: 0.0046 | f1分数: 88.2783 | time: 1290.3442
# Step: 11700 | train loss: 0.1792 | f1分数: 87.0673 | time: 1295.8957
# Step: 11750 | train loss: 0.0234 | f1分数: 88.5397 | time: 1301.4249
# Step: 11800 | train loss: 0.0189 | f1分数: 83.7938 | time: 1306.9492
# Step: 11850 | train loss: 0.1141 | f1分数: 83.6094 | time: 1312.4846
# Step: 11900 | train loss: 0.0326 | f1分数: 86.9307 | time: 1317.9675
# Step: 11950 | train loss: 0.0612 | f1分数: 88.2253 | time: 1323.5122
# Step: 12000 | train loss: 0.0129 | f1分数: 88.6848 | time: 1329.0459
# Step: 12050 | train loss: 0.0119 | f1分数: 89.1130 | time: 1334.6089
# Step: 12100 | train loss: 0.0320 | f1分数: 89.3910 | time: 1340.1171
# Step: 12150 | train loss: 0.0027 | f1分数: 88.7880 | time: 1345.6308
# Step: 12200 | train loss: 0.0063 | f1分数: 88.8835 | time: 1351.1214
# Step: 12250 | train loss: 0.0166 | f1分数: 90.0001 | time: 1356.6426
# Step: 12300 | train loss: 0.0009 | f1分数: 90.3925 | time: 1362.2193
# Step: 12350 | train loss: 0.0007 | f1分数: 89.7504 | time: 1367.7354
# Step: 12400 | train loss: 0.0002 | f1分数: 90.2871 | time: 1373.2332
# Step: 12450 | train loss: 0.0006 | f1分数: 89.8600 | time: 1378.7924
# Step: 12500 | train loss: 0.0012 | f1分数: 89.7944 | time: 1384.3617
# Step: 12550 | train loss: 0.0033 | f1分数: 88.1526 | time: 1389.9146
# Step: 12600 | train loss: 0.0293 | f1分数: 89.3423 | time: 1395.4430
# Step: 12650 | train loss: 0.0044 | f1分数: 89.9380 | time: 1401.0073
# Step: 12700 | train loss: 0.0001 | f1分数: 89.9239 | time: 1406.5862
# Step: 12750 | train loss: 0.0007 | f1分数: 89.7991 | time: 1412.1247
# Step: 12800 | train loss: 0.0005 | f1分数: 89.4358 | time: 1417.7154
# Step: 12850 | train loss: 0.0011 | f1分数: 89.8925 | time: 1423.2522
# Step: 12900 | train loss: 0.0155 | f1分数: 89.5676 | time: 1428.7882
# Step: 12950 | train loss: 0.0011 | f1分数: 90.0029 | time: 1434.3598
# Step: 13000 | train loss: 0.0094 | f1分数: 89.5985 | time: 1439.8696
# Step: 13050 | train loss: 0.0240 | f1分数: 90.1384 | time: 1445.4118
# Step: 13100 | train loss: 0.0523 | f1分数: 88.8164 | time: 1450.9446
# Step: 13150 | train loss: 0.0176 | f1分数: 89.1985 | time: 1456.4401
# Step: 13200 | train loss: 0.0432 | f1分数: 87.6375 | time: 1462.0078
# Step: 13250 | train loss: 0.0045 | f1分数: 89.3850 | time: 1467.5643
# Step: 13300 | train loss: 0.0134 | f1分数: 88.8736 | time: 1473.1041
# Step: 13350 | train loss: 0.0032 | f1分数: 89.4431 | time: 1478.6399
# Step: 13400 | train loss: 0.0002 | f1分数: 89.8094 | time: 1484.1812
# Step: 13450 | train loss: 0.0064 | f1分数: 90.0255 | time: 1489.7337
# Step: 13500 | train loss: 0.0138 | f1分数: 90.1913 | time: 1495.2766
# Step: 13550 | train loss: 0.0001 | f1分数: 90.3219 | time: 1500.8095
# Step: 13600 | train loss: 0.0047 | f1分数: 90.1712 | time: 1506.2979
# Step: 13650 | train loss: 0.0001 | f1分数: 90.1280 | time: 1511.8255
# Step: 13700 | train loss: 0.0001 | f1分数: 90.1132 | time: 1517.3554
# Step: 13750 | train loss: 0.0001 | f1分数: 90.1648 | time: 1522.9258
# Step: 13800 | train loss: 0.0001 | f1分数: 90.2085 | time: 1528.4426
# Step: 13850 | train loss: 0.0003 | f1分数: 90.3503 | time: 1533.9796
# Step: 13900 | train loss: 0.0002 | f1分数: 90.1410 | time: 1539.5547