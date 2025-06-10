import numpy as np
import json

# === Activation functions ===
def relu(x):
    # Implement the Rectified Linear Unit
    return np.maximum(0, x)

def softmax(x):
    # Implement the SoftMax function that works with both 1D and 2D arrays
    # If input is 1D, make it 2D temporarily
    x_ndim = x.ndim
    if x_ndim == 1:
        x = x.reshape(1, -1)

    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    result = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Return to original dimensions
    if x_ndim == 1:
        return result.flatten()
    return result

# === Flatten ===
def flatten(x):
    # Make sure this works with single samples too
    if x.ndim == 3:  # Single sample (1, height, width)
        return x.reshape(x.shape[0], -1)
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return np.dot(x, W) + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data

    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer.get('config', {})
        wnames = layer.get('weights', [])

        # Print debug info
        # print(f"Processing layer: {lname}, type: {ltype}, x shape: {x.shape}")

        if ltype == "Flatten":
            x = flatten(x)
            # print(f"After flatten: {x.shape}")

        elif ltype == "Dense" and wnames:
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)

            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)
            # print(f"After dense+{activation}: {x.shape}")

        elif ltype in ["BatchNormalization", "Dropout"]:
            # Skip these layers during inference
            # print(f"Skipping {ltype} layer")
            continue

    return x

# Main inference function
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)