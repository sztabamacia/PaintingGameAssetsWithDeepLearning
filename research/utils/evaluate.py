


from collections import OrderedDict
import tensorflow as tf
import numpy as np
import json


def eval_regression(X, y, yhat, log, path, bs=32):
    print('Evaluating...')

    X = X[..., 0:3] # extrai só o rascunho
    X = tf.convert_to_tensor(X, tf.float32)
    y = tf.convert_to_tensor(y, tf.float32)
    yhat = tf.convert_to_tensor(yhat, tf.float32)
    yhat = tf.clip_by_value(yhat, 0, 255)

    # Métricas tradicionais
    result = OrderedDict()
    for k, v in log.items():
        result[k] = v

    def compute(a, b, metric):
        # Execução por batches das métricas. Evita OOM no cálculo do SSIM e FID
        result = [metric(a[s:s+bs], b[s:s+bs]) for s in range(0, len(a), bs)]
        result = tf.concat(result, axis=0)

        if len(tf.shape(result)) > 1: # convertendo para 'por imagem', não 'por pixel'
            result = tf.reduce_mean(result, axis=(1, 2))

        return float(tf.reduce_mean(result)), float(tf.math.reduce_std(result))

    def register_metric(metric, name, unbounded):
        result[f'baseline_{name}'], result[f'baseline_{name}_std'] = (b, _) = compute(X, y, metric)
        result[f'raw_{name}'], result[f'raw_{name}_std'] = (r, _) = compute(y, yhat, metric)
        result[f'normalized_{name}'] = r / (b + 1e-7) if unbounded else (r - b) / (1 - b + 1e-7)

    register_metric(tf.keras.losses.mae, 'MAE', True)
    register_metric(lambda a, b: tf.sqrt(tf.reduce_mean(tf.keras.losses.mse(a, b), axis=(1,2))), 'RMSE', True)
    register_metric(lambda a, b: tf.image.ssim(a, b, 255), 'SSIM', False)
    register_metric(lambda a, b: tf.image.ssim_multiscale(a, b, 255), 'MS_SSIM', False)

    for i, (k, v) in enumerate(result.items()):
        print(f' -{k}: {v:0.2f}')
        if i > len(log) and (i - len(log) + 1) % 5 == 0: 
            print()

    if path != None:
        with open(path, 'w') as f:
            json.dump(result, f, indent=4)


from sklearn import metrics
def evaL_classification(y, yhat, log, path, bs=32):
    shape = y.shape
    channels = np.shape(y)[-1]
    y = y.reshape((-1, channels))
    yhat = yhat.reshape((-1, channels))

    # Tirando background da conta
    mask = y[:, 0] == 0
    y, yhat = y[mask], yhat[mask]

    # Tirando classes sem representatividade
    mask = np.sum(y, axis=0) > 0
    y, yhat = y[:, mask], yhat[:, mask]
    yhat = np.nan_to_num(yhat)

    t = np.argmax(y, axis=-1)
    p = np.argmax(yhat, axis=-1)

    print(metrics.classification_report(t, p))
    result = OrderedDict()
    result['N'] = y.shape[0]
    result['C'] = y.shape[1]
    result['acc'] = metrics.accuracy_score(t, p)
    result['balanced_acc'] = metrics.balanced_accuracy_score(t, p)
    result['topk=3'] = metrics.top_k_accuracy_score(t, yhat, k=3)
    result['topk=5'] = metrics.top_k_accuracy_score(t, yhat, k=5)
    result['report'] = metrics.classification_report(t, p, output_dict=True)
    result['confusion'] = metrics.confusion_matrix(t, p, normalize='true').tolist()

    # filepath: c:\studia\pg\metody_badawcze_w_inf\...\evaluate.py
    if np.isnan(yhat).any():
        print("Warning: NaN values found in predictions.")
    yhat = np.nan_to_num(yhat)
    for i, (k, v) in enumerate(result.items()):
        if k == 'report':
            break
        
        print(f' -{k}: {v}')
        if i > len(log) and (i - len(log) + 1) % 5 == 0: 
            print()

    if path != None:
        with open(path, 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    X = tf.zeros((8, 256, 256, 3), dtype=tf.float32)
    y = tf.zeros_like(X)
    yhat = tf.zeros_like(X)
    r = eval(X, y, yhat, None)
    print('Done!')