

import numpy as np
import argparse
import json
import time
import os

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print('\nTensorFlow 2 is ready!\n\n\n\n')

from utils.data_loader import make_dataset
from utils.evaluate import eval_regression, evaL_classification
from utils.models import UNet
import utils.sprites as sprites
import utils.helper as helper

# Configurable parameters
# These are all exposed as command line arguments and use the following values as defaults
tasks_name = 'entcom'   # simple label to track experiements
character = 'Saulo'     # the character to process
input = 'lineart'       # the input sprites to use (from ./dataset/)
output = 'regions'      # the sprite style the network will produce (from ./dataset/)
epochs = 200            # number of training epochs
batch_size = 8          # tune to the highest value that does not get you an out-of-memory exception
n_filters = 72          # 72 was used for the paper. Higher might lead to better results, at the cost of more GPU memory
dense = 2               # 0 = normal classification, 1 = dense supervision, 2 = dense supervision with progressive weights
multires = 1            # 0 = regular Conv-Bn-Relu blocks, 1 = densely connected blocks
loss = 'wfl'            # cce, fl or wfl (categorical cross entropy and (weighted) focal loss)
learning_rate = 0.001   # RMSprop learning rate. We currently do not emply any LR scheduling algorithm
load_previous_model = 0 # whether to search for a previously trained model and use it as starting point
debug = 0               # if 1, the dataset is truncated to just 8 sprites for faster testing


# Command line arguments setup
parser = argparse.ArgumentParser()
parser.add_argument('-t',  '--task', dest='tasks_name', type=str, default=tasks_name)
parser.add_argument('-c',  '--character', dest='character', type=str, default=character)
parser.add_argument('-i',  '--input', dest='input', type=str, default=input)
parser.add_argument('-o',  '--output', dest='output', type=str, default=output)
parser.add_argument('-e',  '--epochs', dest='epochs', type=int, default=epochs)
parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=batch_size)
parser.add_argument('-nf', '--n_filters', dest='n_filters', type=int, default=n_filters)
parser.add_argument('-ds', '--dense', dest='dense', type=int, default=dense)
parser.add_argument('-mr', '--multires', dest='multires', type=int, default=multires)
parser.add_argument('-l',  '--loss', dest='loss', type=str, default=loss)
parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=learning_rate)
parser.add_argument('-lm', '--load_previous_model', dest='load_previous_model', type=int, default=load_previous_model)
parser.add_argument('-db', '--debug', dest='debug', type=int, default=debug)

# We "unpack" the parsed arguments back to the global namespace
args = parser.parse_args().__dict__
for key in args.keys():
    globals()[key] = args[key]



# Creates the model folder and loads the dataset
model_name = f'{character}_{tasks_name}{n_filters}_{input}2{output}.h5'
model_path = f'./results/{time.strftime("%Y %m %d - %H %M %S")}_{model_name[:-3]}/'
X, meta = sprites.load_character(character, input)
y, meta = sprites.load_character(character, output)
if debug > 0:
    X, y, meta = X[::8], y[::8], meta[::8]

input_palette = sprites.extract_palette(X)
output_palette = sprites.extract_palette(y)
X_train, y_train, meta_train, X_test, y_test, meta_test = sprites.train_test_split_by_animation(X, y, meta)
del X, y, meta # saves memory

print('Converting to one-hot encoding...')
X_train = sprites.to_onehot(X_train, input_palette)
y_train = sprites.to_onehot(y_train, output_palette)
X_test = sprites.to_onehot(X_test, input_palette)
y_test = sprites.to_onehot(y_test, output_palette)

model = UNet(X_train, y_train, n_filters, [1, 2, 4, 8, 8, 8, 8], 'relu', 'softmax', dense > 0, multires > 0)
if load_previous_model == 1:
    model = helper.load_weights(model, './results/*/', model_name)

if loss == 'cce':
    loss = tf.keras.losses.CategoricalCrossentropy()
elif loss == 'fl':
    loss = helper.focal_loss(1, 2)
elif loss == 'wfl':
    loss = helper.weighted_focal_loss(y_train)

model.compile(
    tf.keras.optimizers.RMSprop(learning_rate), loss, 
    loss_weights=[7/64, 7/64, 7/32, 7/16, 7/8, 7/4, 7/2] if dense > 1 else None)


# Saves all currently used parameters to disk along with the model to ease experiments tracking
os.makedirs(model_path, exist_ok=True)
with open(model_path + model_name + '_args.json', 'w') as f:
    json.dump(args, f, indent=4)

# Function called each epoch to document the learning progress.
# Each 10 epochs, output classification and regression metrics
# If "epoch" is a string, dump the algorithms outputs too
def evaluate_model(epoch, log):
    epoch = epoch + 1 if type(epoch) == int else epoch
    validate = epoch % 10 == 0 if type(epoch) == int else True
    dump = type(epoch) == str

    i = np.random.choice(np.arange(len(X_test)), size=6)
    y_hat = model.predict(X_test[i], batch_size=batch_size)
    sprites.create_grid([
        sprites.from_onehot(X_test[i], input_palette), 
        sprites.from_onehot(y_hat, output_palette),
        sprites.from_onehot(y_test[i], output_palette)], 
        f'{model_path}{epoch}.png')

    if validate:
        print()
        y_hat1 = model.predict(X_test[:len(X_test)//2], batch_size=batch_size)
        y_hat2 = model.predict(X_test[len(X_test)//2:], batch_size=batch_size)
        y_hat = np.concatenate([y_hat1, y_hat2], axis=0)
        _ = eval_regression(
            sprites.from_onehot(X_test, input_palette), 
            sprites.from_onehot(y_test, output_palette), 
            sprites.from_onehot(y_hat, output_palette), 
            log, f'{model_path}{epoch}_regression.json')
        _ = evaL_classification(y_test, y_hat, log, f'{model_path}{epoch}_classification.json')

    if dump:
        y_hat1 = model.predict(X_test[:len(X_test)//2], batch_size=batch_size)
        y_hat2 = model.predict(X_test[len(X_test)//2:], batch_size=batch_size)
        y_hat = np.concatenate([y_hat1, y_hat2], axis=0)
        sprites.dump([
            sprites.from_onehot(X_test, input_palette), 
            sprites.from_onehot(y_test, output_palette), 
            sprites.from_onehot(y_hat, output_palette)], 
            meta_test, f'{model_path}{epoch}/')


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_path + model_name, save_weights_only=True),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_model)
]

if epochs > 0:
    history = model.fit(
        make_dataset(X_train, y_train, batch_size),
        validation_data=make_dataset(X_test, y_test, batch_size), 
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, use_multiprocessing=False)
    with open(model_path + model_name + '_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

# Avaliação final do modelo
evaluate_model('final_results', {})
