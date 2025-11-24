"""Script para probar predicciones del modelo entrenado."""
from Main import load_model, predict_single, evaluate_on_folder, CONFIDENCE_THRESHOLD
from pathlib import Path

if __name__ == "__main__":
    print("Cargando modelo entrenado...")
    models = load_model("cnn_model.npz")
    print("Modelo cargado exitosamente.\n")
    
    # Probar con imágenes de predicción
    print("=" * 60)
    print("EVALUANDO IMÁGENES DE PREDICCIÓN")
    print("=" * 60)
    evaluate_on_folder("imagenesPrediccion", *models, threshold=CONFIDENCE_THRESHOLD)
    
    # También probar con imágenes de entrenamiento para verificar
    print("\n" + "=" * 60)
    print("VERIFICANDO CON IMÁGENES DE ENTRENAMIENTO")
    print("=" * 60)
    for i in range(1, 6):
        img_path = Path("Images") / f"{i}.png"
        if img_path.exists():
            result = predict_single(str(img_path), *models, threshold=CONFIDENCE_THRESHOLD, verbose=True)

