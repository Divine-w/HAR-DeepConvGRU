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

def conv1d(xs):
    xs = tf.reshape(xs, [-1, n_steps, n_input, 1])
    conv1 = tf.layers.conv2d(xs, 32, [5, 1], [2, 1], 'valid')
    conv2 = tf.layers.conv2d(conv1, 32, [5, 1], [2, 1], 'valid')
    conv3 = tf.layers.conv2d(conv2, 32, [5, 1], [2, 1], 'valid')
    conv4 = tf.layers.conv2d(conv3, 32, [5, 1], [2, 1], 'valid')
    shape = conv4.get_shape().as_list()
    print('dense input shape: {}'.format(shape))
    flat = tf.reshape(conv4, [-1, shape[1] * shape[2] * shape[3]])
    fc1 = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1, units=32, activation=tf.nn.relu)
    output = tf.layers.dense(fc2, n_classes, activation=tf.nn.softmax)  # output based on the last output step

    return output

xs = tf.placeholder(tf.float32, [None, n_steps, n_input],name='input')
ys = tf.placeholder(tf.float32, [None, n_classes],name='label')

output = conv1d(xs)

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
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: batch_xs, ys: batch_ys, })
        train_losses.append(loss_)
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, feed_dict={xs: X_test, ys: y_test})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_
                  , '| time: %.4f' % (time.time() - start_time))
        if step % 100 == 0 and op == 1 and accuracy_>min_acc:
            saver.save(sess, "./ConvLSTMmodel/ConvLSTM_model")
            min_acc = accuracy_
            print('ConvLSTM模型保存成功')
        if step % 1000 == 0:
            indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
            plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
            plt.show()
        step += 1

# Step: 50 | train loss: 0.4427 | test accuracy: 0.7720 | time: 38.8474
# Step: 100 | train loss: 0.1783 | test accuracy: 0.8527 | time: 79.7253
# Step: 150 | train loss: 0.0960 | test accuracy: 0.8741 | time: 118.3365
# Step: 200 | train loss: 0.1257 | test accuracy: 0.8711 | time: 155.0124
# Step: 250 | train loss: 0.1201 | test accuracy: 0.8734 | time: 192.0655
# Step: 300 | train loss: 0.0484 | test accuracy: 0.8860 | time: 228.6203
# Step: 350 | train loss: 0.0675 | test accuracy: 0.8765 | time: 265.2282
# Step: 400 | train loss: 0.0815 | test accuracy: 0.8592 | time: 302.8187
# Step: 450 | train loss: 0.0976 | test accuracy: 0.8653 | time: 339.2684
# Step: 500 | train loss: 0.0976 | test accuracy: 0.8714 | time: 375.2298
# Step: 550 | train loss: 0.0572 | test accuracy: 0.8633 | time: 411.7926
# Step: 600 | train loss: 0.0845 | test accuracy: 0.8968 | time: 447.9191
# Step: 650 | train loss: 0.0682 | test accuracy: 0.8890 | time: 484.3868
# Step: 700 | train loss: 0.1401 | test accuracy: 0.8870 | time: 520.8356
# Step: 750 | train loss: 0.0892 | test accuracy: 0.8904 | time: 559.2256
# Step: 800 | train loss: 0.0466 | test accuracy: 0.8914 | time: 596.5410
# Step: 850 | train loss: 0.0248 | test accuracy: 0.8911 | time: 632.9577
# Step: 900 | train loss: 0.0437 | test accuracy: 0.8880 | time: 670.1999
# Step: 950 | train loss: 0.0702 | test accuracy: 0.8860 | time: 706.4805
# Step: 1000 | train loss: 0.1239 | test accuracy: 0.8833 | time: 743.2435