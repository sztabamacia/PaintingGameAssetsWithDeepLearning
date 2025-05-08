

import tensorflow as tf
import numpy as np



def UNet(input, output, n_filters, multipliers=[1,2,4,8], default_activation='relu', output_activation='linear', dense=True, multires=True):      
    def block(n_filters, size, stride):
        """Convolution - BatchNormalization - Activation trio"""
        def _operation(flow):
            flow = tf.keras.layers.Conv2D(n_filters, size, stride, 'SAME', activation='linear', use_bias=False)(flow)
            flow = tf.keras.layers.BatchNormalization()(flow)
            flow = tf.keras.layers.Activation(activation=default_activation)(flow)
            return flow
        return _operation

    def multiresblock(n_filters):
        """3 Conv-Bn-Relu blocks in sequence and parallel + residual connection"""
        def _operation(flow):
            a = flow
            b = flow = block(int(np.round(n_filters / 3)), 3, 1)(flow)
            c = flow = block(int(np.round(n_filters / 3)), 3, 1)(flow)
            d = flow = block(int(np.round(n_filters / 3)), 3, 1)(flow)
            concat = flow = tf.keras.layers.Concatenate()([b, c, d])
            if concat.shape[-1] != a.shape[-1]:
                a = block(n_filters, 1, 1)(a)
            return tf.keras.layers.Add()([a, concat])
        return _operation

    def downsample(n_filters):
        """
        Basic downsampling step. 
        When dense supervision is enabled, each downsampling step receives a downsampled version of the input
        When dense connections is enabled, three blocks are used to create a similar number of weights
        """
        def _operation(flow, downsampled_input):
            if downsampled_input is not None:
                flow = tf.keras.layers.Concatenate()([flow, downsampled_input])
            if multires:
                flow = multiresblock(n_filters)(flow)
                flow = multiresblock(n_filters)(flow)
                flow = multiresblock(n_filters)(flow)
            else:
                flow = block(n_filters, 3, 1)(flow)
                flow = block(n_filters, 3, 1)(flow)
            flow = tf.keras.layers.MaxPooling2D()(flow)
            return flow
        return _operation

    def upsample(n_filters):
        def _operation(flow, skip):
            if multires:
                flow = multiresblock(n_filters)(flow)
                flow = multiresblock(n_filters)(flow)
                flow = multiresblock(n_filters)(flow)
                flow = tf.keras.layers.UpSampling2D(interpolation='nearest')(flow)
                flow = multiresblock(n_filters//2)(flow)
                flow = tf.keras.layers.Concatenate()([flow, multiresblock(n_filters//2)(multiresblock(n_filters//2)(skip))])
            else:
                flow = block(n_filters, 3, 1)(flow)
                flow = block(n_filters, 3, 1)(flow)
                flow = tf.keras.layers.UpSampling2D(interpolation='nearest')(flow)
                flow = block(n_filters//2, 3, 1)(flow)
                flow = tf.keras.layers.Concatenate()([flow, skip])
            return flow
        return _operation

    def classifier(n_filters, size, stride):
        """Creates a classification block"""
        def _operation(flow):
            return tf.keras.layers.Conv2D(n_filters, size, stride, activation=output_activation, padding='same', dtype=tf.float32, name=f'out{flow.shape[1]}')(flow)
        return _operation


    input_shape = input.shape[1:]
    output_channels = output.shape[-1]
    
    flow = imp = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    down = [flow]
    downsampled_input = flow if dense else None
    for m in multipliers:
        flow = downsample(n_filters * m)(flow, downsampled_input)
        downsampled_input = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(downsampled_input) if dense else None        
        down.append(flow)

    # Encoder
    flow = down.pop()
    outputs = []
    for (d, m) in zip(down[::-1], multipliers[::-1]):
        flow = upsample(n_filters * m)(flow, d)
        if dense:
            outputs.append(classifier(output_channels, 3, 1)(flow))
    if not dense:
        outputs = classifier(output_channels, 3, 1)(flow)


    model = tf.keras.models.Model(inputs=imp, outputs=outputs, name='unet_' + str(n_filters))
    # Sobreescrevendo algumas funções do model para que ele lide automaticamente com o 'dense'
    if dense:
        _train_step = model.train_step
        _test_step = model.test_step
        _predict = model.predict
        def expand_labels(y):
            labels = [y]
            for _ in range(6, 0, -1):
                labels.append(tf.nn.avg_pool2d(labels[-1], 2, 2, 'SAME'))
            return labels[::-1]
        def train_step(data):
            X, y = data
            return _train_step((X, expand_labels(y)))
        def test_step(data):
            X, y = data
            return _test_step((X, expand_labels(y)))
        def predict(*args, **kwargs):
            return _predict(*args, **kwargs)[-1]

        model.train_step = train_step
        model.test_step = test_step
        model.predict = predict
    return model

if __name__ == "__main__":
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    input = np.random.uniform(low=0, high=1, size=(8, 256, 256, 3)).astype(np.float32)
    output = np.random.uniform(low=0, high=1, size=(8, 256, 256, 42)).astype(np.float32)
    for dense in [False, True]:
        for multires in [False, True]:
            model = UNet(input, output, 72, [1,2,4,8,8,8,8], 'relu', 'softmax', dense, multires)
            print(f'{dense} {multires} {model.count_params()}')
            #model.compile('adam', 'categorical_crossentropy', ['acc'])
            #model.run_eagerly = True
            #model.summary()
            #model.fit(input, output, validation_data=(input, output), epochs=1, batch_size=len(input))
            #yhat = model.predict(input)
            #del model

    print('All cases working!')