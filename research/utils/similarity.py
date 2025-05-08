


from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from PIL import Image
import tensorflow as tf
import numpy as np

def find_similar(X, meta, k=5, allow_same_anim=True):
    preprocessed = tf.keras.applications.vgg16.preprocess_input(X.copy())
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, pooling='avg')
    vgg.trainable = False    
    features = vgg.predict(preprocessed, batch_size=128)
    neighbors = NearestNeighbors(n_neighbors=len(X)//5, radius=2, metric='cosine').fit(features)
    indices = neighbors.kneighbors(features)[1][1:,:]

    # Remove 'similares' que vem da mesma animação que a própria imagem
    if not allow_same_anim:
        indices = [[int(idx) for idx in index if meta[i][1] != meta[idx][1]] for (i, index) in enumerate(indices)]
    # Seleciona só os 'k' pares mais parecidos
    indices = [index[:k] for index in indices]

    return indices

def make_pairs(X, y, meta, indices):
    pairs = [[(i, idx) for idx in index] for (i, index) in enumerate(indices)]
    pairs = sum(pairs, [])
    p1, p2 = list(zip(*pairs))
    p1, p2 = list(p1), list(p2)
    return np.concatenate([X[p1], y[p2]], axis=-1), y[p1], [meta[i] for i in p1]

def debug_similar(X, indices, take_only=16):
    for i, idx in enumerate(indices):
        row = np.hstack(np.vstack([X[i:i+1], X[idx]]))
        Image.fromarray(row.astype(np.uint8)).save(f'./debug/{i}.png')
        if i > take_only:
            break