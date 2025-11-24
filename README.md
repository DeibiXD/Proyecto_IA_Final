# Red Neuronal Convolucional desde Cero

Implementación completa de una Red Neuronal Convolucional (CNN) para clasificación de imágenes desde cero, sin usar librerías de alto nivel como TensorFlow, Keras o PyTorch.

## Descripción

Este proyecto implementa una CNN completa para clasificar imágenes en 5 clases diferentes. La red incluye:
- Capas convolucionales con backpropagation
- Capas de pooling (MaxPool)
- Capas fully connected
- Funciones de activación (ReLU, tanh)
- Algoritmo de entrenamiento con descenso de gradiente

## Características

-  Implementación desde cero (solo NumPy y PIL)
-  Backpropagation manual para todas las capas
-  Data augmentation para mejorar generalización
-  Guardado y carga de modelos entrenados
-  Sistema de predicción con umbral de confianza
-  Código modular y organizado
## Requisitos

### Software
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Librerías Python
```bash
pip install numpy pillow
```

O instalar desde `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   cd Intento_3
   ```

2. **Instalar dependencias**
   ```bash
   pip install numpy pillow
   ```

3. **Organizar las imágenes de entrenamiento**
   
   Coloca tus imágenes en las carpetas correspondientes:
   - `Images/Uno/` - Imágenes de la clase 1
   - `Images/Dos/` - Imágenes de la clase 2
   - `Images/Tres/` - Imágenes de la clase 3
   - `Images/Four/` - Imágenes de la clase 4
   - `Images/Five/` - Imágenes de la clase 5
   
   **Formatos soportados**: `.png`, `.jpg`, `.jpeg`
   
   **Recomendación**: Mínimo 20-50 imágenes por clase para buenos resultados.

## Uso

### 1. Entrenar el Modelo

> **Nota**: El proceso de entrenamiento puede tomar entre 50-70 minutos dependiendo de tu hardware. Con early stopping, puede terminar antes si no hay mejora. Por cuestiones de tiempo no se proporciona un modelo pre-entrenado, pero el código está listo para entrenar.

Ejecuta el script principal para entrenar la red:

```bash
python Main.py
```

El script:
- Carga todas las imágenes de las carpetas en `Images/`
- Aplica data augmentation (flip horizontal, ruido gaussiano, ajustes de brillo)
- Normaliza los datos
- Entrena la red neuronal
- Guarda el modelo en `cnn_model.npz`

**Parámetros de entrenamiento** (editar en `Main.py`):
- `epochs`: Número de épocas (default: 400, con early stopping)
- `initial_lr`: Learning rate inicial (default: 0.1)
- `CONFIDENCE_THRESHOLD`: Umbral de confianza para predicciones (default: 0.2)

### 2. Hacer Predicciones

#### Opción A: Usando el script de prueba
```bash
python test_predictions.py
```

Este script:
- Carga el modelo entrenado
- Evalúa todas las imágenes en `imagenesPrediccion/`
- Muestra resultados detallados

#### Opción B: Desde Python interactivo
```python
from Main import load_model, predict_single

# Cargar modelo entrenado
models = load_model("cnn_model.npz")

# Predecir una imagen
resultado = predict_single(
    "imagenesPrediccion/number-one.png", 
    *models, 
    threshold=0.2,  # Umbral de confianza
    verbose=True    # Mostrar detalles
)

print(resultado)
# Output:
# {
#     'label': 'Clase_1',
#     'confidence': 0.85,
#     'raw_probs': array([0.85, 0.05, 0.03, 0.04, 0.03])
# }
```

#### Opción C: Evaluar carpeta completa
```python
from Main import load_model, evaluate_on_folder

models = load_model("cnn_model.npz")
evaluate_on_folder("imagenesPrediccion", *models, threshold=0.2)
```

## Arquitectura de la Red

La CNN implementada tiene la siguiente arquitectura:

```
Input (28x28x1)
    ↓
Conv2D (8 filtros 3x3) + ReLU
    ↓
MaxPool 2x2 → (13x13x8)
    ↓
Flatten → (1352)
    ↓
FullyConnected (64) + tanh
    ↓
FullyConnected (5) + Softmax
    ↓
Output (5 clases)
```

### Detalles de la Arquitectura

1. **Conv2D(1→8 filtros 3x3) + ReLU**: Los filtros pequeños capturan bordes y texturas básicas con pocos parámetros. ReLU evita saturación en la parte convolucional y mantiene gradientes grandes.

2. **MaxPool 2x2**: Reduce la resolución a 13x13 y aporta invariancia local. Reduce la dimensionalidad espacial a la mitad.

3. **FullyConnected 8×13×13 → 64 con tanh**: Combina las características globales extraídas por las capas convolucionales. La función tanh mantiene acotadas las activaciones previas a la capa de salida.

4. **FullyConnected 64 → 5 + Softmax**: Capa de salida que produce probabilidades sobre las 5 clases mediante softmax.

### Funciones de Activación

- **ReLU**: Utilizada en la parte convolucional para evitar saturación temprana y acelerar el aprendizaje. Mantiene gradientes grandes y evita el problema de gradientes que desaparecen.
- **tanh**: Utilizada en la capa densa intermedia para mantener las activaciones acotadas antes de la capa de salida.

### Preprocesamiento

- **Normalización**: Todas las imágenes se normalizan restando la media del dataset y dividiendo por la desviación estándar. Esto asegura que los datos estén centrados y escalados apropiadamente.
- **Data Augmentation**: Se aplica augmentación adaptativa según el tamaño del dataset:
  - Con pocas imágenes (<20 por clase): Flip horizontal, ruido gaussiano y ajustes de brillo
  - Con muchas imágenes (≥20 por clase): Solo flip horizontal para optimizar velocidad
  - Esto genera aproximadamente 2x más ejemplos de entrenamiento (con datasets grandes)

### Estrategia de Entrenamiento

- **Learning Rate Decay**: El learning rate se reduce gradualmente:
  - Épocas 0-150: Learning rate inicial (0.1)
  - Épocas 150-250: Learning rate × 0.5
  - Épocas 250+: Learning rate × 0.25
- **Early Stopping**: Se detiene automáticamente si no hay mejora por 100 épocas (después de la época 150). Esto previene sobreajuste y ahorra tiempo de cómputo.

### Estrategia "Ninguna de las Anteriores"

Se evalúa el máximo de softmax; si es menor al umbral de confianza (default: 0.2) se reporta 'Ninguna'. El umbral puede ajustarse según validación para equilibrar falsos positivos y negativos. Esto permite detectar imágenes que no pertenecen a ninguna de las 5 clases de entrenamiento.

## Formato de Salida

Las predicciones retornan un diccionario con:

```python
{
    'label': 'Clase_1' o 'Ninguna',
    'confidence': 0.85,  # Probabilidad de la clase predicha
    'raw_probs': array([0.85, 0.05, 0.03, 0.04, 0.03])  # Probabilidades de todas las clases
}
```

## Configuración Avanzada

### Ajustar Hiperparámetros

Edita `Main.py` en la función `train()`:

```python
def train(epochs=400, initial_lr=0.1):
    # Cambiar número de épocas
    # Cambiar learning rate inicial
    # Ajustar arquitectura (número de filtros, neuronas, etc.)
```

### Cambiar Umbral de Confianza

```python
# En Main.py
CONFIDENCE_THRESHOLD = 0.2  # Ajustar según necesidad

# O al hacer predicción
predict_single(..., threshold=0.3)  # Más estricto
predict_single(..., threshold=0.15)  # Más permisivo
```

