

import tensorflow as tf
import numpy as np



def make_dataset(X, y, batch_size):
    def random_flip_augmentation(X, y):
        s = tf.shape(X)[-1]
        m = tf.concat([X, y], axis=-1)
        m = tf.image.random_flip_left_right(m)
        m = tf.cast(m, dtype=tf.float32)
        return m[..., :s], m[..., s:]

    with tf.device('/:CPU:0'):
        X = tf.convert_to_tensor(X, dtype=tf.int8)
        y = tf.convert_to_tensor(y, dtype=tf.int8)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(X))
        dataset = dataset.map(random_flip_augmentation) if type(y) != list else dataset
        dataset = dataset.batch(batch_size)
        
    return dataset


# Crude unit test
if __name__ == '__main__':
    X = np.zeros((64, 320, 320, 3), np.uint8)
    y = np.zeros((64, 320, 320, 3), np.uint8)

    dataset = make_dataset(X, y, 32)
    for batch in dataset.as_numpy_iterator():
        print(batch)
        break
    print('Done!')