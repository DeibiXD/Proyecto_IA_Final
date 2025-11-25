import numpy as np
from PIL import Image
from pathlib import Path

from Activations import ReLU
from Layers import Conv2D, FullyConnected, MaxPool2x2

np.random.seed(42)

CLASS_NAMES = [f"Clase_{i}" for i in range(1, 6)]
NONE_OF_ABOVE_LABEL = "Ninguna"
CONFIDENCE_THRESHOLD = 0.2
DATA_MEAN = 0.0
DATA_STD = 1.0

ARCHITECTURE_DOC = "Ver README.md para documentación completa de la arquitectura, preprocesamiento y estrategias de entrenamiento."

# --- data loading helpers ----------------------------------------------------
def load_images(folder="Images", size=(28, 28)):
    """Carga imágenes desde carpetas organizadas por clase.
    
    Estructura esperada:
    Images/
        Uno/    -> Clase 0 (Clase_1)
        Dos/    -> Clase 1 (Clase_2)
        Tres/   -> Clase 2 (Clase_3)
        Four/   -> Clase 3 (Clase_4)
        Five/   -> Clase 4 (Clase_5)
    """
    X = []
    y = []
    folder_path = Path(folder)
    
    # Mapeo de carpetas a clases
    class_folders = {
        0: "Uno",    # Clase_1
        1: "Dos",    # Clase_2
        2: "Tres",   # Clase_3
        3: "Four",   # Clase_4
        4: "Five"    # Clase_5
    }
    
    for class_idx, folder_name in class_folders.items():
        class_folder = folder_path / folder_name
        if not class_folder.exists():
            print(f"Advertencia: carpeta {class_folder} no existe, saltando clase {class_idx+1}")
            continue
        
        # Cargar todas las imágenes de la carpeta
        image_files = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
        
        if len(image_files) == 0:
            print(f"Advertencia: no se encontraron imágenes en {class_folder}")
            continue
        
        print(f"Cargando {len(image_files)} imágenes de {folder_name} (Clase_{class_idx+1})")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("L").resize(size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(class_idx)
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
                continue
    
    print(f"\nTotal de imágenes cargadas: {len(X)}")
    return np.array(X), np.array(y)

def augment_dataset(X, y, max_augmentations=3):
    """Generate augmented data with flip, noise, and brightness adjustments.
    
    Args:
        max_augmentations: Maximum number of augmentation types to apply.
                          Reduces augmentation for faster training when dataset is large.
    """
    augmented_X = [X]
    augmented_y = [y]
    
    # Reduce augmentation for larger datasets to speed up training
    avg_samples_per_class = len(X) / 5
    if avg_samples_per_class > 20:
        # With 100+ images, minimal augmentation is sufficient
        max_augmentations = 1
    
    # Horizontal flip (always useful and fast)
    flipped = np.flip(X, axis=2)
    augmented_X.append(flipped)
    augmented_y.append(y)
    
    if max_augmentations >= 2:
        # Gaussian noise
        noisy = np.clip(X + np.random.normal(0, 0.05, X.shape), 0.0, 1.0)
        augmented_X.append(noisy)
        augmented_y.append(y)
    
    if max_augmentations >= 3:
        bright = np.clip(X * 1.15, 0.0, 1.0)
        augmented_X.append(bright)
        augmented_y.append(y)
    
    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    return X_aug, y_aug


def one_hot(y, num_classes=5):
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

# --- utilities ---------------------------------------------------------------
def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(pred, target):
    eps = 1e-7
    return -np.sum(target * np.log(pred + eps)) / target.shape[0]

# --- inference helpers -------------------------------------------------------
def forward_pass(X, conv, relu, pool, fc, out_layer):
    z1 = conv.forward(X)
    a1 = relu.forward(z1)
    p1 = pool.forward(a1)
    flat = p1.reshape(p1.shape[0], -1)
    hidden_lin = fc.forward(flat)
    hidden = np.tanh(hidden_lin)
    logits = out_layer.forward(hidden)
    probs = softmax(logits)
    cache = {
        "p1": p1,
        "flat": flat,
        "hidden": hidden,
        "hidden_lin": hidden_lin,
    }
    return probs, cache

def predict_single(image_path, conv, relu, pool, fc, out_layer, threshold=CONFIDENCE_THRESHOLD, verbose=False):
    """Predict class for a single image."""
    img = Image.open(image_path).convert("L").resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    normalized = (arr - DATA_MEAN) / (DATA_STD + 1e-6)
    tensor = normalized[None, None, :, :]
    probs, _ = forward_pass(tensor, conv, relu, pool, fc, out_layer)
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[0, max_idx])
    label = CLASS_NAMES[max_idx] if max_prob >= threshold else NONE_OF_ABOVE_LABEL
    
    result = {"label": label, "confidence": max_prob, "raw_probs": probs.flatten()}
    
    if verbose:
        print(f"\nPredicción para {Path(image_path).name}:")
        print(f"  Clase predicha: {label}")
        print(f"  Confianza: {max_prob:.4f}")
        print(f"  Probabilidades por clase:")
        for i, prob in enumerate(probs.flatten()):
            print(f"    {CLASS_NAMES[i]}: {prob:.4f}")
    
    return result

def evaluate_on_folder(folder_path, conv, relu, pool, fc, out_layer, threshold=CONFIDENCE_THRESHOLD):
    """Evaluate model on all images in a folder."""
    folder = Path(folder_path)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    
    print(f"\nEvaluando {len(image_files)} imágenes en {folder_path}:")
    print("=" * 60)
    
    for img_path in image_files:
        predict_single(str(img_path), conv, relu, pool, fc, out_layer, threshold, verbose=True)

def save_model(path, conv, relu, pool, fc, out_layer):
    """Persist trained weights, biases y estadísticos de normalización."""
    np.savez(
        path,
        conv_W=conv.W,
        conv_b=conv.b,
        conv_lr=conv.lr,
        conv_kernel=conv.kernel_size,
        conv_in=conv.W.shape[1],
        conv_out=conv.out_channels,
        fc_W=fc.W,
        fc_b=fc.b,
        fc_lr=fc.lr,
        fc_in=fc.W.shape[0],
        fc_out=fc.W.shape[1],
        out_W=out_layer.W,
        out_b=out_layer.b,
        out_lr=out_layer.lr,
        out_in=out_layer.W.shape[0],
        out_out=out_layer.W.shape[1],
        data_mean=DATA_MEAN,
        data_std=DATA_STD,
    )

def load_model(path):
    """Reconstruye las capas con pesos guardados previamente."""
    global DATA_MEAN, DATA_STD
    data = np.load(path, allow_pickle=True)

    conv = Conv2D(
        in_channels=int(data["conv_in"]),
        out_channels=int(data["conv_out"]),
        kernel_size=int(data["conv_kernel"]),
        lr=float(data["conv_lr"]),
    )
    conv.W = data["conv_W"]
    conv.b = data["conv_b"]

    fc_mid = FullyConnected(
        in_units=int(data["fc_in"]),
        out_units=int(data["fc_out"]),
        lr=float(data["fc_lr"]),
    )
    fc_mid.W = data["fc_W"]
    fc_mid.b = data["fc_b"]

    out_layer = FullyConnected(
        in_units=int(data["out_in"]),
        out_units=int(data["out_out"]),
        lr=float(data["out_lr"]),
    )
    out_layer.W = data["out_W"]
    out_layer.b = data["out_b"]

    DATA_MEAN = float(data["data_mean"])
    DATA_STD = float(data["data_std"])

    relu = ReLU()
    pool = MaxPool2x2()
    return conv, relu, pool, fc_mid, out_layer

# --- training loop -----------------------------------------------------------
def train(epochs=200, initial_lr=0.1):
    global DATA_MEAN, DATA_STD
    import time
    start_time = time.time()
    
    X, y = load_images()
    load_time = time.time()
    print(f"Tiempo de carga: {load_time - start_time:.2f}s")
    
    X, y = augment_dataset(X, y)
    aug_time = time.time()
    print(f"Tiempo de augmentación: {aug_time - load_time:.2f}s")
    
    DATA_MEAN = X.mean()
    DATA_STD = X.std() + 1e-6
    X = (X - DATA_MEAN) / DATA_STD
    X = X[:, None, :, :]  # add channel dimension
    targets = one_hot(y)

    print(f"Dataset aumentado: {X.shape[0]} muestras")
    print(f"Media: {DATA_MEAN:.4f}, Desv. Est.: {DATA_STD:.4f}\n")

    conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, lr=initial_lr)
    relu = ReLU()
    pool = MaxPool2x2()
    fc = FullyConnected(in_units=8 * 13 * 13, out_units=64, lr=initial_lr)
    out_layer = FullyConnected(64, 5, lr=initial_lr)

    best_acc = 0.0
    best_loss = float('inf')
    patience = 100
    no_improve = 0

    for epoch in range(epochs):
        if epoch < 150:
            current_lr = initial_lr
        elif epoch < 250:
            current_lr = initial_lr * 0.5
        else:
            current_lr = initial_lr * 0.25
        
        conv.lr = current_lr
        fc.lr = current_lr
        out_layer.lr = current_lr

        probs, cache = forward_pass(X, conv, relu, pool, fc, out_layer)
        p1 = cache["p1"]
        hidden = cache["hidden"]

        loss = cross_entropy(probs, targets)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        
        # Track both accuracy and loss
        improved = False
        if acc > best_acc + 0.01 or (acc >= best_acc and loss < best_loss - 0.01):
            best_acc = acc
            best_loss = loss
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        if epoch % 50 == 0 or improved:
            print(f"epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.5f}, best_acc={best_acc:.4f}")

        if no_improve >= patience and epoch > 150:
            print(f"\nEarly stopping en epoch {epoch} (sin mejora por {patience} épocas)")
            break

        grad_logits = (probs - targets) / targets.shape[0]
        grad_hidden = out_layer.backward(grad_logits)
        grad_hidden *= (1 - hidden ** 2)  # tanh derivative
        grad_flat = fc.backward(grad_hidden)
        grad_pool = grad_flat.reshape(p1.shape)
        grad_relu = pool.backward(grad_pool)
        grad_conv = relu.backward(grad_relu)
        conv.backward(grad_conv)

    total_time = time.time() - start_time
    print(f"\nMejor accuracy alcanzada: {best_acc:.4f}, mejor loss: {best_loss:.4f}")
    print(f"Tiempo total de entrenamiento: {total_time:.2f}s ({total_time/60:.2f} minutos)")
    return conv, relu, pool, fc, out_layer

if __name__ == "__main__":
    models = train()
    save_model("cnn_model.npz", *models)
    print("\nModelo guardado en cnn_model.npz")
    print("\n" + "=" * 60)
    print(ARCHITECTURE_DOC.strip())
    print("=" * 60)
    
    # Opcional: evaluar en imágenes de predicción
    print("\n¿Deseas evaluar el modelo en imágenes de predicción? (descomenta las líneas siguientes)")
    # models = load_model("cnn_model.npz")
    # evaluate_on_folder("imagenesPrediccion", *models, threshold=CONFIDENCE_THRESHOLD)