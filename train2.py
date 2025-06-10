import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import json

# 確保有 model 目錄
os.makedirs('model', exist_ok=True)

# 載入 Fashion-MNIST 資料集
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 標準化數據到 [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 創建模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.1
)

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 保存模型為所需的格式
def save_model_for_testing(model, base_path):
    # 創建模型架構
    model_arch = []
    weights_dict = {}

    # 首先添加 Flatten 層
    flatten_layer = {
        "name": "flatten",
        "type": "Flatten",
        "config": {},
        "weights": []
    }
    model_arch.append(flatten_layer)

    # 處理 Dense 層
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for i, layer in enumerate(dense_layers):
        layer_name = f"dense_{i+1}"

        # 獲取權重
        W, b = layer.get_weights()
        weight_name = f"{layer_name}_W"
        bias_name = f"{layer_name}_b"

        # 保存權重到字典
        weights_dict[weight_name] = W
        weights_dict[bias_name] = b

        # 確定激活函數
        activation = "linear"
        if layer.activation == tf.keras.activations.relu:
            activation = "relu"
        elif layer.activation == tf.keras.activations.softmax:
            activation = "softmax"

        # 創建層配置
        layer_config = {
            "name": layer_name,
            "type": "Dense",
            "config": {"activation": activation},
            "weights": [weight_name, bias_name]
        }

        model_arch.append(layer_config)

    # 保存架構為 JSON
    json_path = f"{base_path}.json"
    with open(json_path, 'w') as f:
        json.dump(model_arch, f)

    # 保存權重為 NPZ
    npz_path = f"{base_path}.npz"
    np.savez(npz_path, **weights_dict)

    print(f"模型架構已保存到: {json_path}")
    print(f"模型權重已保存到: {npz_path}")

# 保存模型
save_model_for_testing(model, 'model/fashion_mnist')
