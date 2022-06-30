

from collections import defaultdict, namedtuple
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import numpy as np
import json
import os


SpriteInfo = namedtuple("SpriteInfo", "Character AnimationID AnimationFrame xOffset yOffset")
def load_character(character, tag, crop=(32, 32), root='./dataset/'):
    """
    Loads all the sprites for a given character and tag, crops them, and returns the images and their
    metadata
    
    :param character: The name of the character to load
    :param tag: The subset of sprites to load (e.g., encoded, regions, shading, etc.)
    :param crop: The amount of pixels to crop from the top, bottom, left, and right of the image
    :param root: the root directory of the dataset, defaults to ./dataset/ (optional)
    :return: Images and meta
    """
    sprites = glob(f'{root}{tag}/{character}*.png')
    meta = [os.path.basename(s)[:-4].split('_') for s in sprites]
    meta = [SpriteInfo(c, int(aid), int(af), int(xo), int(yo)) for (c, aid, af, xo, yo) in meta]
    if len(sprites) == 0:
        raise Exception(f'No sprite was found for {character} using the {tag} tag.')
        
    images = [np.array(Image.open(s))[crop[0]:-crop[0], crop[1]:-crop[1]] for s in sprites]
    images = np.stack(images, axis=0)

    return images, meta

def extract_palette(sprites):
    """
    Takes a set of sprites and returns all unique colors used
    
    :param sprites: a numpy array of shape (..., 3)
    :return: The unique colors
    """
    pixels = sprites.reshape((-1, 3))
    pixels = pixels[(pixels != (0,0,0)).all(axis=-1)]
    palette = np.unique(pixels, axis=0) # too slow if we don't prune background pixels first
    palette = np.vstack([palette, [0,0,0]])
    return palette

def train_test_split_by_animation(X, y, meta):
    """
    Splits the data into training and test sets, ensuring that the first 3/4ths of each animation
    belongs to train and the last 1/4th belongs to test
    In a sense, this split evaluates whether the algorithm can generalize to new frames of known
    animations. In other words, if it could work as an auto-complete engine.
    This is more favorable than adding different animations to train and test, as this
    approach would evaluate the model's creativity to new animations, not its capacity to help
    artists get their job done faster.
    
    :param X: the sprites to split
    :param y: the labels for each sprite
    :param meta: a list of metadata objects, one for each sprite
    :return: the X_train, y_train, meta_train, X_test, y_test, and meta_test objects
    """
    sprites_by_animation = defaultdict(lambda: [])
    for i, m in enumerate(meta):
        sprites_by_animation[m.AnimationID].append(i)

    train, test = [], []
    for a in sprites_by_animation.values():
        if len(a) <= 2:
            train += a
        else:
            split = int(len(a) * 0.75)
            train += a[:split]
            test += a[split:]
    
    meta_train = [m for (i, m) in enumerate(meta) if i in train]
    meta_test = [m for (i, m) in enumerate(meta) if i in test]
    return X[train], y[train], meta_train, X[test], y[test], meta_test

def create_grid(images, path):
    lines = [np.hstack(np.clip(line, 0, 255).astype(np.uint8)) for line in images]
    mosaic = np.vstack(lines)
    Image.fromarray(mosaic).save(path)

def dump(images, meta, path):
    images = [np.clip(i, 0, 255).astype(np.uint8) for i in images]
    images = np.dstack(images)
    os.makedirs(path, exist_ok=True)
    for row, m in zip(images, meta):
        name = f'{m.Character}_{m.AnimationID}_{m.AnimationFrame}.png'
        Image.fromarray(row).save(path + name)

def to_onehot(X, palette):
    """
    Converts the set of RGB sprites into one-hot encoded sprites following the palette.
    Works for single images and multiple images
    
    :param X: the input sprites
    :param palette: a list of RGB colors
    """
    *dims, c = np.shape(X)
    X = np.clip(X, 0, 255).astype(np.uint8)
    onehot = np.zeros((*dims, len(palette)), np.int8)
    for i, color in enumerate(palette):
        onehot[..., i] = (X == color).all(axis=-1)

    return onehot

def from_onehot(X, palette):
    """
    It takes the argmax of the last axis of the input array, and uses that as an index into the palette
    array
    
    :param X: the image to be converted
    :param palette: a list of RGB colors, e.g. [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    :return: The index of the maximum value in the array.
    """
    X = np.argmax(X, axis=-1)
    X = palette[X]
    return X

def quantize(X, palette):
    """
    It takes an image and a palette, and returns the image with each pixel replaced by the closest color
    in the palette
    
    :param X: the image to be quantized
    :param palette: a list of RGB colors
    :return: The reconstructed image.
    """ 
    X, shape = np.clip(X, 0, 255), np.shape(X)
    X = X.reshape((-1, 3))
    distances = np.sum(np.subtract.outer(X, palette.transpose()) ** 2, axis=(1, 2))
    reconstructed = np.argmin(distances, axis=-1)
    reconstructed = palette[reconstructed]
    reconstructed = reconstructed.reshape(shape)
    return reconstructed


# Crude unit tests
if __name__ == '__main__':
    sprites, meta = load_character('Chris', 'shading')
    sprites, meta = sprites[0:32], meta[0:32] # just to debug faster
    palette = extract_palette(sprites)

    onehot = to_onehot(sprites, palette)
    original = from_onehot(onehot, palette)
    assert (sprites == original).all()

    salt = np.random.randint(-10, 10, sprites.shape)
    reconstructed = quantize(sprites.astype(np.float) + salt, palette)
    assert np.mean(sprites == reconstructed) > 0.75

    create_grid([sprites[0:4], sprites[4:8], sprites[8:12]], "./debug/debug_mosaic.png")
    dump(sprites[0:12], meta[0:12], './debug/')

    X_train, y_test, meta_train, X_test, y_test, meta_test = train_test_split_by_animation(sprites, sprites, meta)

    print('Tests complete!')
