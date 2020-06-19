import numpy as np
import pickle as cp
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 64

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 32

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

assert NB_SENSOR_CHANNELS == X_train.shape[1]
def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def one_hot(y_, n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]

# Sensor data is segmented using a sliding window mechanism
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (training): inputs {0}, targets {1}\n".format(X_train.shape, y_train.shape),
      " ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

y_train = one_hot(y_train, NUM_CLASSES)
y_test = one_hot(y_test, NUM_CLASSES)
print("train_x shape: ", X_train.shape, "| train_y (labels) shape: ", y_train.shape
      , "| test_x shape: ", X_test.shape, "| test_y (labels) shape: ", y_test.shape)

print('是否将处理后的数据保存为numpy文件：1 是 2 否 ')
sel = int(input('请选择： '))
if sel == 1:
    np.save("./Opp_64/processed/np_train_x.npy", X_train)
    np.save("./Opp_64/processed/np_train_y.npy", y_train)
    np.save("./Opp_64processed/np_test_x.npy", X_test)
    np.save("./Opp_64/processed/np_test_y.npy", y_test)
    print('数据已保存为numpy文件')

# Data is reshaped since the input of the network is a 4 dimension tensor
X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))