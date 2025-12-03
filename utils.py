import os
import io
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown 

# Constantes
MODEL_PATH = "ckpt_finetuned.pt"
CLASSES_PATH = os.path.join("meta", "classes.txt")
DRIVE_ID = "10lXt4B9W6ZFBoUALMpCFxoWcRsT8jLWB" 

def download_model_if_missing():
    """Descarga el modelo automáticamente si no existe en la carpeta"""
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ El modelo {MODEL_PATH} no existe. Descargando desde Google Drive...")
        print("   (Esto puede tardar unos minutos la primera vez...)")
        url = f'https://drive.google.com/uc?id={DRIVE_ID}'
        # quiet=False para ver la barra de progreso en los logs
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Descarga completada.")
    else:
        print("✅ Modelo encontrado localmente.")

def _load_classes(path: str = CLASSES_PATH) -> List[str]:
    # Tu lógica personalizada de clases (21 clases)
    if not os.path.exists(path):
        # Fallback de seguridad por si no existe el txt
        print(f"⚠️ Alerta: No se encontró {path}, usando lista vacía.")
        return ["unknown"] * 21
        
    with open(path, "r", encoding="utf-8") as f:
        classes = f.read().splitlines()
    # Tu lógica específica: las primeras 20 + "other"
    classes_21 = classes[:20] + ["other"]
    return classes_21

def _build_model(device: torch.device, checkpoint_path: str):
    # --- TU ARQUITECTURA EXACTA (DenseNet201) ---
    print(f"🏗️ Construyendo arquitectura DenseNet201 personalizada...")
    backbone = models.densenet201(weights=None)

    # Classifier: 1920 -> 1024 -> 101
    classifier = nn.Sequential(
        nn.Linear(1920, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 101),
    )
    backbone.classifier = classifier

    # Head: 101 -> 21
    # Nota: Aquí llamamos a _load_classes para saber el tamaño de salida (21)
    head = nn.Linear(101, len(_load_classes()))

    # Complete model
    model = nn.Sequential(backbone, head)

    # Cargar checkpoint forzando CPU (vital para Render/Docker)
    print(f"📂 Cargando pesos desde {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if isinstance(state, dict):
        if 'model_state_dict' in state:
            sd = state['model_state_dict']
        elif 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
    else:
        sd = state

    # Limpieza de claves (quitar "module.")
    sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"⚠️ Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"⚠️ Warning: Unexpected keys: {unexpected[:5]}...")

    model.to(device)
    model.eval()
    return model

_test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _test_transforms(img).unsqueeze(0)

def load_model_and_classes(checkpoint_path: str = MODEL_PATH) -> Tuple[torch.nn.Module, List[str], torch.device]:
    """Función principal que llama app.py"""
    
    # 1. AUTO-DESCARGA (Nuevo)
    download_model_if_missing()
    
    # 2. Configuración de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Verificación final
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    # 4. Construcción y carga
    model = _build_model(device, checkpoint_path)
    classes = _load_classes()
    
    print("🚀 Modelo cargado y listo para inferencia.")
    return model, classes, device

def predict_from_bytes(image_bytes: bytes, model: torch.nn.Module, classes: List[str], device: torch.device, topk: int = 3):
    """Return top-k predictions as list of (label, probability)."""
    tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze(0)

    topk_probs, topk_idx = torch.topk(probs, k=topk)
    results = []
    for p, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
        label = classes[idx] if idx < len(classes) else "unknown"
        results.append({"label": label, "confidence": float(p)})

    return results