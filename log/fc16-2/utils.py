import numpy as np
import math
import torch

def load_torch_data(file_name):
    points_radar_with_anno = np.loadtxt(file_name, delimiter=',')
    '''
    radar points channels: 
    x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms label
    '''
    train_ratio = 0.8
    num_train = math.floor(points_radar_with_anno.shape[0] * train_ratio)
    # train data
    X_train_orig = points_radar_with_anno[:num_train, :18]
    # delete channels: 
    #   z / rcs / is_quality_valid / ambig_state / invalid_state / pdh0 / vx_rms / vy_rms
    X_train_orig = np.delete(X_train_orig, [2,10,11,14,15,16,17], axis=1)
    Y_train_orig = points_radar_with_anno[:num_train, -1]
    Y_train_orig[Y_train_orig > 1] = 1
    # test data
    X_test_orig = points_radar_with_anno[num_train:, :18]
    X_test_orig = np.delete(X_test_orig, [2,10,11,14,15,16,17], axis=1)
    Y_test_orig = points_radar_with_anno[num_train:, -1]
    Y_test_orig[Y_test_orig > 1] = 1
    print("Trainset ground ratio: ", 1.-1.*np.sum(Y_train_orig)/Y_train_orig.shape[0])
    print("Testset ground ratio: ", 1.-1.*np.sum(Y_test_orig)/Y_test_orig.shape[0])
    # to torch tensor
    X_train = torch.from_numpy(
        X_train_orig / np.max(X_train_orig, axis=1).reshape((-1, 1))).type(torch.FloatTensor) #
    X_test = torch.from_numpy(
        X_test_orig / np.max(X_test_orig, axis=1).reshape((-1, 1))).type(torch.FloatTensor) #
    Y_train = torch.from_numpy(Y_train_orig).type(torch.int64)
    Y_test = torch.from_numpy(Y_test_orig).type(torch.int64)
    print("X_train:", X_train.shape, X_train.dtype,)
    print("Y_train:", Y_train.shape, Y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("Y_test:", Y_test.shape, Y_test.dtype)

    # set weights according to num of each label
    weights,_ = np.histogram(Y_train, range(3))
    weights = weights.astype(np.float32) / np.sum(weights)
    weights = np.power(np.amax(weights) / weights, 1 / 1.0)
    print('label weights:', weights)

    return X_train, Y_train, X_test, Y_test, weights

def load_tf_data(file_name):
    points_radar_with_anno = np.loadtxt(file_name, delimiter=',')
    train_ratio = 0.8
    num_train = math.floor(points_radar_with_anno.shape[0] * train_ratio)
    X_train_orig = points_radar_with_anno[:num_train, :18].T
    Y_train_orig = points_radar_with_anno[:num_train, -1].astype(int)
    Y_train_orig[Y_train_orig > 1] = 1

    X_test_orig = points_radar_with_anno[num_train:, :18].T
    Y_test_orig = points_radar_with_anno[num_train:, -1].astype(int)
    Y_test_orig[Y_test_orig > 1] = 1

    return X_train_orig, Y_train_orig, X_test_orig, Y_test_orig

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches