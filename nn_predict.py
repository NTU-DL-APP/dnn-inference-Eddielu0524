import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    # 減去最大值，避免指數爆炸
    exp_x = np.exp(x - max_x)
    # 計算和
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    # 返回 softmax 結果
    return exp_x / sum_exp_x

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = x.reshape(x.shape[0], -1)
        elif ltype == "Dense":
            if len(wnames) >= 2:
                # 獲取權重和偏置
                W = weights[wnames[0]]
                b = weights[wnames[1]]

                # 線性變換
                x = np.dot(x, W) + b

                # 應用激活函數
                activation = cfg.get("activation", "linear")
                if activation == "relu":
                    x = relu(x)
                elif activation == "softmax":
                    x = softmax(x)

        elif ltype in ["BatchNormalization", "Dropout"]:
            # 推理時跳過這些層
            continue

    return x


# You are free to replace nn_forward_h5() with your own implementation
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)