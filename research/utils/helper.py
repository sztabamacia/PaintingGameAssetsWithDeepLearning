

from glob import glob
import tensorflow as tf
import numpy as np
import os


def load_model(search_path, model_name):
    print('Loading previous model...')
    model = None
    previous_models = glob(search_path + model_name)
    if len(previous_models) > 0:
        most_recent = max(previous_models, key=os.path.getctime)
        print(' -Model found: ' + most_recent)
        try:
            model = tf.keras.models.load_model(most_recent)
        except BaseException as e:
            print(' -Failed to load model. Reason: ' + str(e))
    else:
        print(' -No previous model found to load :x')

    return model

def load_weights(model, search_path, model_name):
    print('Loading previous model...')
    previous_models = glob(search_path + model_name)
    if len(previous_models) > 0:
        most_recent = max(previous_models, key=os.path.getctime)
        print(' -Model found: ' + most_recent)
        try:
            model.load_weights(most_recent)
        except BaseException as e:
            print(' -Failed to load weights. Reason: ' + str(e))
    else:
        print(' -No previous model found to load :x')

    return model

# Beta pequeno = equilibrado, beta grande = foco nas classes menos presentes
def class_weights(y, beta=0.9):
    n_classes = y.shape[-1]
    u, counts = np.unique(np.argmax(y, axis=-1), return_counts=True)
    
    counts = counts / np.sum(counts)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * n_classes
    class_weights = np.zeros((n_classes,))
    class_weights[u] = weights
    
    return class_weights

def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        loss = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * loss  # focal loss
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fixed

def weighted_focal_loss(y, beta=0.9, gamma=2.0):
    weights = class_weights(y, beta)
    weights = tf.convert_to_tensor(weights, tf.float32)
    return focal_loss(weights, gamma)

