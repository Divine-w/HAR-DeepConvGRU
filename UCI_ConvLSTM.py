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
num_units_lstm = 32
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

def ConvLSTM(xs):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
    conv1 = tf.layers.conv2d(xs, 32, [5, 1], [2, 1], 'valid')
    conv2 = tf.layers.conv2d(conv1, 32, [5, 1], [2, 1], 'valid')
    conv3 = tf.layers.conv2d(conv2, 32, [5, 1], [2, 1], 'valid')
    conv4 = tf.layers.conv2d(conv3, 32, [5, 1], [2, 1], 'valid')
    shape = conv4.get_shape().as_list()
    print('gru input shape: {}'.format(shape))
    flat = tf.reshape(conv4, [-1, shape[1], shape[2] * shape[3]])
    rnn_cell_1 = tf.contrib.rnn.LSTMCell(num_units_lstm)
    rnn_cell_2 = tf.contrib.rnn.LSTMCell(num_units_lstm)
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

output = ConvLSTM(xs)

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

# Step: 50 | train loss: 0.9074 | test accuracy: 0.6223 | time: 4.5905
# Step: 100 | train loss: 0.3303 | test accuracy: 0.7883 | time: 7.8352
# Step: 150 | train loss: 0.2827 | test accuracy: 0.8568 | time: 11.0970
# Step: 200 | train loss: 0.1745 | test accuracy: 0.8697 | time: 14.3767
# Step: 250 | train loss: 0.1994 | test accuracy: 0.8945 | time: 17.6855
# Step: 300 | train loss: 0.0556 | test accuracy: 0.8985 | time: 20.9042
# Step: 350 | train loss: 0.0978 | test accuracy: 0.8921 | time: 24.2929
# Step: 400 | train loss: 0.0987 | test accuracy: 0.8965 | time: 27.4877
# Step: 450 | train loss: 0.1795 | test accuracy: 0.8965 | time: 30.8414
# Step: 500 | train loss: 0.1494 | test accuracy: 0.8996 | time: 34.1262
# Step: 550 | train loss: 0.0396 | test accuracy: 0.9016 | time: 37.3379
# Step: 600 | train loss: 0.0623 | test accuracy: 0.9016 | time: 40.7296
# Step: 650 | train loss: 0.0756 | test accuracy: 0.9002 | time: 43.9854
# Step: 700 | train loss: 0.1246 | test accuracy: 0.9013 | time: 47.3641
# Step: 750 | train loss: 0.1384 | test accuracy: 0.8989 | time: 50.6129
# Step: 800 | train loss: 0.0452 | test accuracy: 0.9013 | time: 53.9106
# Step: 850 | train loss: 0.0344 | test accuracy: 0.8918 | time: 57.1993
# Step: 900 | train loss: 0.0678 | test accuracy: 0.8968 | time: 60.5711
# Step: 950 | train loss: 0.1209 | test accuracy: 0.8968 | time: 63.9219
# Step: 1000 | train loss: 0.1268 | test accuracy: 0.8904 | time: 67.3366

# Step: 50 | train loss: 0.6743 | test accuracy: 0.6189 | time: 4.7201
# Step: 100 | train loss: 0.4468 | test accuracy: 0.7879 | time: 8.1918
# Step: 150 | train loss: 0.3720 | test accuracy: 0.8317 | time: 11.5386
# Step: 200 | train loss: 0.1787 | test accuracy: 0.8761 | time: 14.8053
# Step: 250 | train loss: 0.1753 | test accuracy: 0.8850 | time: 18.1911
# Step: 300 | train loss: 0.0497 | test accuracy: 0.8833 | time: 21.6609
# Step: 350 | train loss: 0.0757 | test accuracy: 0.8996 | time: 25.1237
# Step: 400 | train loss: 0.1048 | test accuracy: 0.8914 | time: 28.5454
# Step: 450 | train loss: 0.1879 | test accuracy: 0.8931 | time: 31.6311
# Step: 500 | train loss: 0.1682 | test accuracy: 0.9186 | time: 34.8729
# Step: 550 | train loss: 0.0347 | test accuracy: 0.9206 | time: 38.2026
# Step: 600 | train loss: 0.1420 | test accuracy: 0.9186 | time: 41.7084
# Step: 650 | train loss: 0.0643 | test accuracy: 0.9125 | time: 44.9841
# Step: 700 | train loss: 0.1190 | test accuracy: 0.9108 | time: 48.3479
# Step: 750 | train loss: 0.1343 | test accuracy: 0.9152 | time: 51.6276
# Step: 800 | train loss: 0.0364 | test accuracy: 0.9186 | time: 54.9074
# Step: 850 | train loss: 0.0434 | test accuracy: 0.9091 | time: 58.2251
# Step: 900 | train loss: 0.0710 | test accuracy: 0.9121 | time: 61.5348
# Step: 950 | train loss: 0.1212 | test accuracy: 0.9097 | time: 64.7546
# Step: 1000 | train loss: 0.1626 | test accuracy: 0.9050 | time: 68.0553

# Retro
# Step: 50 | train loss: 0.6791 | test accuracy: 0.7007 | time: 41.6084
# Step: 100 | train loss: 0.3235 | test accuracy: 0.8096 | time: 83.8752
# Step: 150 | train loss: 0.2074 | test accuracy: 0.8392 | time: 124.5179
# Step: 200 | train loss: 0.2147 | test accuracy: 0.8595 | time: 167.5372
# Step: 250 | train loss: 0.2148 | test accuracy: 0.8809 | time: 208.0448
# Step: 300 | train loss: 0.0550 | test accuracy: 0.8660 | time: 248.8266
# Step: 350 | train loss: 0.0559 | test accuracy: 0.8687 | time: 291.4817
# Step: 400 | train loss: 0.0721 | test accuracy: 0.8687 | time: 332.0793
# Step: 450 | train loss: 0.1406 | test accuracy: 0.8795 | time: 375.2098
# Step: 500 | train loss: 0.1108 | test accuracy: 0.8751 | time: 417.6807
# Step: 550 | train loss: 0.0278 | test accuracy: 0.8782 | time: 459.2160
# Step: 600 | train loss: 0.0322 | test accuracy: 0.8761 | time: 500.5992
# Step: 650 | train loss: 0.0675 | test accuracy: 0.8636 | time: 542.0185
# Step: 700 | train loss: 0.1294 | test accuracy: 0.8748 | time: 585.7584
# Step: 750 | train loss: 0.1095 | test accuracy: 0.8823 | time: 635.3021
# Step: 800 | train loss: 0.0257 | test accuracy: 0.8918 | time: 679.4413
# Step: 850 | train loss: 0.0560 | test accuracy: 0.8850 | time: 723.9867
# Step: 900 | train loss: 0.0396 | test accuracy: 0.8839 | time: 767.7566
# Step: 950 | train loss: 0.0882 | test accuracy: 0.8860 | time: 811.4614
# Step: 1000 | train loss: 0.0481 | test accuracy: 0.8856 | time: 854.3787