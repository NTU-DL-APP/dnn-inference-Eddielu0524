import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os

# 設置隨機種子確保可重現性
np.random.seed(42)
tf.random.set_seed(42)

# 1. 載入 Fashion-MNIST 資料集
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 標準化數據到 [0, 1]
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 2. 創建一個高性能模型
model = Sequential([
    # 將 28x28 圖像展平
    Flatten(input_shape=(28, 28)),

    # 第一個全連接層
    Dense(784, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # 第二個全連接層
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # 第三個全連接層
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # 輸出層 - 10個類別
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 定義回調函數
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# 3. 訓練模型
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 4. 評估最終模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 5. 保存模型為 h5 格式（可選）
model.save('fashion_mnist.h5')

# 6. 轉換模型為 JSON 和 NPZ 格式
def convert_model_to_json_npz(model, json_path, npz_path):
    # 創建模型架構的 JSON 表示
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

    # 處理所有 Dense 層
    dense_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_count += 1
            layer_name = f"dense_{dense_count}"

            # 獲取權重和偏置
            W, b = layer.get_weights()
            weight_name = f"{layer_name}_W"
            bias_name = f"{layer_name}_b"

            # 保存權重到字典
            weights_dict[weight_name] = W
            weights_dict[bias_name] = b

            # 創建層配置
            activation = "linear"
            if layer.activation == tf.keras.activations.relu:
                activation = "relu"
            elif layer.activation == tf.keras.activations.softmax:
                activation = "softmax"

            layer_config = {
                "name": layer_name,
                "type": "Dense",
                "config": {"activation": activation},
                "weights": [weight_name, bias_name]
            }

            model_arch.append(layer_config)

    # 保存架構為 JSON
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(model_arch, f, indent=2)

    # 保存權重為 NPZ
    np.savez(npz_path, **weights_dict)

    print(f"Model architecture saved to: {json_path}")
    print(f"Model weights saved to: {npz_path}")

# 創建 model 目錄（如果不存在）
os.makedirs('model', exist_ok=True)

# 轉換並保存模型
convert_model_to_json_npz(
    model,
    'model/fashion_mnist.json',
    'model/fashion_mnist.npz'
)


