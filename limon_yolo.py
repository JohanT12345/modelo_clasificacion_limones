"""
=============================================================
  Pipeline YOLOv8 - Clasificación de madurez Limón Tahití
  Clases: exportacion | consumo_interno | desecho
=============================================================
"""

import os
import shutil
import random
import yaml
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

# ─────────────────────────────────────────────
#  ① CONFIGURACIÓN - CAMBIA SOLO ESTA SECCIÓN
# ─────────────────────────────────────────────

# Ruta a tu carpeta con las 3 subcarpetas de clases
RUTA_ORIGINAL = r"dataset_original"  

# Dónde se creará todo el dataset preparado
RUTA_DATASET  = r"limon_dataset"      

# Cuántas imágenes generar por clase con augmentation
IMAGENES_POR_CLASE = 300

# Proporciones train / val / test
SPLIT = {"train": 0.70, "val": 0.20, "test": 0.10}

# Clases en el mismo orden que tus carpetas
CLASES = ["exportacion", "consumo_interno", "desecho"]

# Parámetros de entrenamiento
EPOCHS   = 100
IMG_SIZE = 640
MODELO   = "yolov8n-cls.pt"  

# ─────────────────────────────────────────────
#   DATA AUGMENTATION
# ─────────────────────────────────────────────

def construir_augmentador():
    """
    Pipeline de augmentation enfocado en variaciones reales de campo:
    - Iluminación variable (sol, sombra, túnel de cosecha)
    - Orientación aleatoria (limones en distintas posiciones)
    - Ruido y desenfoque (cámaras de baja resolución)
    - Cambios de color suaves (simulación de distintas condiciones de luz)
    """
    return A.Compose([
        # Volteos y rotaciones
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),

        # Recortes aleatorios
        A.RandomResizedCrop(
            height=640, width=640,
            scale=(0.75, 1.0),
            ratio=(0.8, 1.2),
            p=0.5
        ),

        # Brillo, contraste y color (simula distintas luces de campo)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=25,
                p=1.0
            ),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.8),

        # Simulación de condiciones de imagen reales
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.4),

        # Desenfoque suave (cámara en movimiento o fuera de foco)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Simulación de sombras y manchas de luz
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), num_flare_circles_lower=1,
                         num_flare_circles_upper=3, src_radius=100, p=0.15),

        # Resize final a tamaño estándar
        A.Resize(height=640, width=640),
    ])


def augmentar_clase(carpeta_clase: Path, carpeta_salida: Path, clase_nombre: str, n_imagenes: int):
    """Genera n_imagenes aumentadas a partir de las fotos originales de una clase."""

    imagenes_originales = list(carpeta_clase.glob("*.jpg")) + \
                          list(carpeta_clase.glob("*.jpeg")) + \
                          list(carpeta_clase.glob("*.png")) + \
                          list(carpeta_clase.glob("*.JPG")) + \
                          list(carpeta_clase.glob("*.JPEG")) + \
                          list(carpeta_clase.glob("*.PNG"))

    if not imagenes_originales:
        raise FileNotFoundError(f"No se encontraron imágenes en: {carpeta_clase}")

    print(f"\n  [{clase_nombre}] {len(imagenes_originales)} fotos originales → generando {n_imagenes} aumentadas")

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    augmentador = construir_augmentador()

    # Copiar primero las originales (redimensionadas)
    for img_path in imagenes_originales:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))
        destino = carpeta_salida / img_path.name
        cv2.imwrite(str(destino), img)

    # Generar imágenes aumentadas hasta alcanzar n_imagenes
    generadas = len(imagenes_originales)
    contador  = 0

    with tqdm(total=n_imagenes - generadas, desc=f"    Augmentando {clase_nombre}", unit="img") as pbar:
        while generadas < n_imagenes:
            img_path = random.choice(imagenes_originales)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultado = augmentador(image=img_rgb)
            img_aug   = cv2.cvtColor(resultado["image"], cv2.COLOR_RGB2BGR)

            nombre_salida = carpeta_salida / f"aug_{clase_nombre}_{contador:05d}.jpg"
            cv2.imwrite(str(nombre_salida), img_aug, [cv2.IMWRITE_JPEG_QUALITY, 92])

            generadas += 1
            contador  += 1
            pbar.update(1)

    print(f"    → {generadas} imágenes guardadas en {carpeta_salida}")


# ─────────────────────────────────────────────
#   CONSTRUCCIÓN DEL DATASET YOLO
# ─────────────────────────────────────────────

def construir_dataset_yolo(carpeta_aumentada: Path, carpeta_dataset: Path):
    print("\n Construyendo estructura de dataset YOLO...")

    # Limpiar si ya existe
    if carpeta_dataset.exists():
        shutil.rmtree(carpeta_dataset)

    for split in SPLIT:
        for clase in CLASES:
            (carpeta_dataset / split / clase).mkdir(parents=True, exist_ok=True)

    for clase in CLASES:
        imagenes = list((carpeta_aumentada / clase).glob("*.jpg")) + \
                   list((carpeta_aumentada / clase).glob("*.png"))
        random.shuffle(imagenes)

        n_total = len(imagenes)
        n_train = int(n_total * SPLIT["train"])
        n_val   = int(n_total * SPLIT["val"])

        splits_imgs = {
            "train": imagenes[:n_train],
            "val":   imagenes[n_train:n_train + n_val],
            "test":  imagenes[n_train + n_val:],
        }

        for split, imgs in splits_imgs.items():
            for img_path in imgs:
                destino = carpeta_dataset / split / clase / img_path.name
                shutil.copy2(img_path, destino)

        print(f"  {clase}: train={len(splits_imgs['train'])} | val={len(splits_imgs['val'])} | test={len(splits_imgs['test'])}")


def crear_yaml(carpeta_dataset: Path):
    """Crea el archivo dataset.yaml requerido por YOLOv8."""
    config = {
        "path":  str(carpeta_dataset),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    len(CLASES),
        "names": CLASES,
    }
    ruta_yaml = carpeta_dataset / "dataset.yaml"
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"\n dataset.yaml creado en: {ruta_yaml}")
    return ruta_yaml


# ─────────────────────────────────────────────
#  ④ ENTRENAMIENTO YOLOV8
# ─────────────────────────────────────────────

def entrenar(ruta_yaml: Path):
    """Entrena el modelo YOLOv8 en modo clasificación."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n ultralytics no está instalado.")
        print("   Corre:  pip install ultralytics")
        return

    print(f"\n Iniciando entrenamiento YOLOv8 clasificación...")
    print(f"   Modelo base : {MODELO}")
    print(f"   Epochs      : {EPOCHS}")
    print(f"   Imagen size : {IMG_SIZE}")
    print(f"   Dataset     : {ruta_yaml}\n")

    modelo = YOLO(MODELO)
    resultados = modelo.train(
        data    = str(ruta_yaml.parent),   
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = 16,
        patience= 20,         
        workers = 0,         
        project = "runs/classify",
        name    = "limon_madurez",
        exist_ok= True,
        plots   = True,      
        verbose = True,
        device=0,
    )
    return resultados


def evaluar(ruta_dataset: Path):
    """Evalúa el modelo entrenado sobre el conjunto de test."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    ruta_modelo = Path("runs/classify/limon_madurez/weights/best.pt")
    if not ruta_modelo.exists():
        print(f"\n  No se encontró el modelo en {ruta_modelo}")
        return

    print(f"\n Evaluando en conjunto de test...")
    modelo = YOLO(str(ruta_modelo))
    metricas = modelo.val(
        data   = str(ruta_dataset),
        split  = "test",
        imgsz  = IMG_SIZE,
        batch  = 16,
        workers= 0,
    )
    return metricas


def inferencia_ejemplo(ruta_dataset: Path):
    """
    Muestra cómo usar el modelo entrenado para predecir una imagen nueva.
    Solo corre si el modelo ya fue entrenado.
    """
    ruta_modelo = Path("runs/classify/limon_madurez/weights/best.pt")
    if not ruta_modelo.exists():
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        return

    # Buscar una imagen del test para demo
    test_imgs = list((ruta_dataset / "test").rglob("*.jpg"))
    if not test_imgs:
        return

    img_demo = test_imgs[0]
    modelo   = YOLO(str(ruta_modelo))
    preds    = modelo.predict(str(img_demo), imgsz=IMG_SIZE, verbose=False)

    print(f"\n DEMO - Predicción sobre: {img_demo.name}")
    for pred in preds:
        probs = pred.probs
        clase_idx  = int(probs.top1)
        confianza  = float(probs.top1conf)
        clase_name = CLASES[clase_idx]
        print(f"   Clase predicha : {clase_name}")
        print(f"   Confianza      : {confianza*100:.1f}%")
        print(f"\n   Top-3 probabilidades:")
        for i in probs.top5[:3]:
            print(f"     {CLASES[int(i)]:<20} {float(probs.data[int(i)])*100:.1f}%")


# ─────────────────────────────────────────────
#  ⑤ MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PIPELINE YOLOV8 - Madurez Limón Tahití")
    print("=" * 60)

    ruta_original = Path(RUTA_ORIGINAL)
    ruta_dataset  = Path(RUTA_DATASET)
    ruta_aumentada = ruta_dataset.parent / "limon_aumentada"

    # Verificar carpetas de entrada
    for clase in CLASES:
        carpeta = ruta_original / clase
        if not carpeta.exists():
            raise FileNotFoundError(
                f"\n No se encontró la carpeta: {carpeta}\n"
                f"   Asegúrate de que RUTA_ORIGINAL apunte a la carpeta correcta\n"
                f"   y que contenga las subcarpetas: {CLASES}"
            )

    # ── Paso 1: Data augmentation ──
    print("\n🔄 PASO 1/4 - Data Augmentation")
    random.seed(42)
    np.random.seed(42)

    for clase in CLASES:
        augmentar_clase(
            carpeta_clase  = ruta_original / clase,
            carpeta_salida = ruta_aumentada / clase,
            clase_nombre   = clase,
            n_imagenes     = IMAGENES_POR_CLASE,
        )

    # ── Paso 2: Construir dataset YOLO ──
    print("\n PASO 2/4 - Construyendo dataset YOLO")
    construir_dataset_yolo(ruta_aumentada, ruta_dataset)

    # ── Paso 3: Crear YAML ──
    print("\n PASO 3/4 - Creando dataset.yaml")
    ruta_yaml = crear_yaml(ruta_dataset)

    # ── Paso 4: Entrenar ──
    print("\n PASO 4/4 - Entrenamiento")
    entrenar(ruta_yaml)

    # ── Evaluación y demo ──
    evaluar(ruta_dataset)
    inferencia_ejemplo(ruta_dataset)

    print("\n" + "=" * 60)
    print("   Pipeline completado")
    print(f"  Modelo guardado en: runs/classify/limon_madurez/weights/best.pt")
    print(f"  Gráficas en:        runs/classify/limon_madurez/")
    print("=" * 60)


if __name__ == "__main__":
    main()