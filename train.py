import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

# 設置隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 模型文件路徑
YOUR_MODEL_NAME = 'fashion_mnist'
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

def create_optimized_model():
    """創建優化的 Fashion-MNIST 分類模型"""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu', name='dense_1'),
        keras.layers.BatchNormalization(name='batch_norm_1'),
        keras.layers.Dropout(0.3, name='dropout_1'),
        keras.layers.Dense(256, activation='relu', name='dense_2'),
        keras.layers.BatchNormalization(name='batch_norm_2'),
        keras.layers.Dropout(0.3, name='dropout_2'),
        keras.layers.Dense(128, activation='relu', name='dense_3'),
        keras.layers.Dropout(0.2, name='dropout_3'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ])

    return model

def train_model():
    """訓練 Fashion-MNIST 模型"""
    print("🚀 開始訓練 Fashion-MNIST 模型...")

    # 載入數據
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # 預處理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"訓練集形狀: {x_train.shape}")
    print(f"測試集形狀: {x_test.shape}")

    # 創建模型
    model = create_optimized_model()

    # 編譯模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 設置回調
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 訓練
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # 評估
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\\n📊 最終測試準確率: {test_accuracy:.4f}")

    # 保存模型
    model.save(TF_MODEL_PATH)
    print(f"✅ 模型已保存: {TF_MODEL_PATH}")

    return model, test_accuracy

def convert_model_to_numpy(model):
    """將 TensorFlow 模型轉換為 NumPy 格式"""
    print("\\n🔄 轉換模型格式...")

    # === 提取權重 ===
    params = {}
    print("🔍 提取權重...")

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            print(f"  處理層: {layer.name}")
            for i, w in enumerate(weights):
                param_name = f"{layer.name}_{i}"
                print(f"    {param_name}: shape={w.shape}")
                params[param_name] = w

    # 保存權重
    np.savez(MODEL_WEIGHTS_PATH, **params)
    print(f"✅ 權重已保存: {MODEL_WEIGHTS_PATH}")

    # === 提取架構 ===
    arch = []
    print("\\n🏗️  提取架構...")

    for layer in model.layers:
        config = layer.get_config()
        info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": config,
            "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
        }
        arch.append(info)
        print(f"  添加層: {layer.name} ({layer.__class__.__name__})")

    # 保存架構
    with open(MODEL_ARCH_PATH, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"✅ 架構已保存: {MODEL_ARCH_PATH}")

def test_numpy_inference():
    """測試 NumPy 推理功能"""
    print("\\n🧪 測試 NumPy 推理...")

    # 載入權重和架構
    weights = np.load(MODEL_WEIGHTS_PATH)
    with open(MODEL_ARCH_PATH) as f:
        architecture = json.load(f)

    print(f"載入權重文件: {len(weights.files)} 個參數")
    print(f"載入架構文件: {len(architecture)} 層")

    # 激活函數
    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def flatten(x):
        return x.reshape(x.shape[0], -1)

    def dense(x, W, b):
        return x @ W + b

    # 前向傳播
    def forward(x):
        for layer in architecture:
            lname = layer['name']
            ltype = layer['type']
            cfg = layer['config']
            wnames = layer['weights']

            if ltype == "Flatten":
                x = flatten(x)
                print(f"  {lname}: {ltype} -> shape: {x.shape}")

            elif ltype == "Dense":
                if len(wnames) >= 2:
                    W = weights[wnames[0]]
                    b = weights[wnames[1]]
                    x = dense(x, W, b)

                    # 應用激活函數
                    activation = cfg.get("activation", "linear")
                    if activation == "relu":
                        x = relu(x)
                    elif activation == "softmax":
                        x = softmax(x)

                    print(f"  {lname}: {ltype}({activation}) -> shape: {x.shape}")

            elif ltype in ["BatchNormalization", "Dropout"]:
                # 推理時跳過這些層
                print(f"  {lname}: {ltype} (跳過)")
                continue

        return x

    # 測試推理
    print("\\n🎯 執行測試推理...")
    dummy_input = np.random.rand(1, 28, 28).astype(np.float32)
    output = forward(dummy_input)

    print(f"\\n📊 輸出概率: {output[0]}")
    print(f"🎯 預測類別: {np.argmax(output, axis=-1)[0]}")
    print(f"🔢 最高概率: {np.max(output):.4f}")

    return True

def main():
    """主函數"""
    print("=" * 50)
    print("🎯 Fashion-MNIST 完整解決方案")
    print("=" * 50)

    try:
        # 檢查是否已有訓練好的模型
        if os.path.exists(TF_MODEL_PATH):
            print(f"📁 發現現有模型: {TF_MODEL_PATH}")
            model = keras.models.load_model(TF_MODEL_PATH)

            # 快速評估
            (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            x_test = x_test.astype('float32') / 255.0
            y_test = keras.utils.to_categorical(y_test, 10)
            _, accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(f"📊 現有模型準確率: {accuracy:.4f}")

        else:
            # 訓練新模型
            model, accuracy = train_model()

        # 轉換模型格式
        convert_model_to_numpy(model)

        # 測試 NumPy 推理
        test_numpy_inference()

        print("\\n" + "=" * 50)
        print("🎉 所有任務完成！")
        print("📁 生成的文件:")
        print(f"  - {TF_MODEL_PATH} (TensorFlow 模型)")
        print(f"  - {MODEL_WEIGHTS_PATH} (NumPy 權重)")
        print(f"  - {MODEL_ARCH_PATH} (JSON 架構)")
        print("=" * 50)

    except Exception as e:
        print(f"❌ 發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
