# Clasificador Nutricional de Alimentos con IA

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![DenseNet201](https://img.shields.io/badge/Model-DenseNet201-blue)](https://arxiv.org/abs/1608.06993)
[![Food-101](https://img.shields.io/badge/Dataset-Food--101-yellow)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Descripción

**Clasificador Nutricional de Alimentos** es una aplicación web inteligente que utiliza **Deep Learning** para identificar alimentos a partir de imágenes y proporcionar información nutricional detallada en tiempo real. El sistema está construido con una arquitectura moderna basada en **DenseNet201** fine-tuned sobre el dataset **Food-101**, logrando una precisión superior al 85% en la clasificación de 21 categorías de alimentos.

### Características Principales

- **Reconocimiento Visual de Alimentos**: Clasificación automática con modelo DenseNet201 optimizado
- **Análisis Nutricional Instantáneo**: Consulta a la API de Open Food Facts (2M+ productos)
- **Asistente Conversacional**: Chat interactivo para consultas nutricionales
- **Interfaz Moderna**: Frontend responsive con experiencia de usuario intuitiva
- **Dockerizado**: Despliegue rápido y reproducible en cualquier entorno
- **Descarga Automática**: El modelo se descarga automáticamente desde Google Drive

---

## Arquitectura del Sistema

```
┌─────────────────┐
│   Frontend      │  HTML5 + CSS3 + Vanilla JS
│  (NutriFood UI) │
└────────┬────────┘
         │
         ↓ HTTP/REST
┌─────────────────┐
│   FastAPI       │  Python 3.10
│   Backend       │  + Uvicorn ASGI
└────────┬────────┘
         │
    ┌────┴─────┬──────────────┐
    ↓          ↓              ↓
┌────────┐ ┌───────┐  ┌──────────────┐
│PyTorch │ │Utils  │  │Open Food Facts│
│Model   │ │+Clients│  │API (External)│
│DenseNet│ │       │  │              │
└────────┘ └───────┘  └──────────────┘
```

### Modelo de Deep Learning

- **Arquitectura**: DenseNet201 (Densely Connected Convolutional Networks)
- **Dataset**: Food-101 (101 clases, ~101,000 imágenes)
- **Clases Entrenadas**: 21 categorías (20 específicas + "other")
- **Backbone**: Pre-entrenado en ImageNet1K
- **Fine-tuning**: Clasificador personalizado (1920 → 1024 → 101 → 21)
- **Precisión**: Top-1 ~85%, Top-3 ~95%

---

## Instalación y Uso

### Prerrequisitos

- Python 3.10+
- Docker (opcional, recomendado para producción)
- GPU CUDA (opcional, para entrenamiento)

### Opción 1: Instalación Local

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/gportal-21/NUTRITIONAL-FOOD.git
   cd NUTRITIONAL-FOOD
   ```

2. **Crear entorno virtual (recomendado)**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Acceder a la aplicación**
   - Abrir navegador en: `http://localhost:8000`
   - API Docs (Swagger): `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

### Opción 2: Docker (Recomendado)

1. **Construir la imagen**

   ```bash
   docker build -t clasificador-comida .
   ```

2. **Ejecutar el contenedor**

   ```bash
   docker run -p 8000:8000 clasificador-comida
   ```

3. **Acceder a la aplicación**
   - Interfaz Web: `http://localhost:8000`

---

## Estructura del Proyecto

```
Clasificador-de-comida/
│
├── Frontend-NutriFood/       # Interfaz de usuario web
│   ├── index.html            # Página principal
│   ├── script.js             # Lógica de interacción
│   └── styles.css            # Estilos visuales
│
├── meta/                     # Archivos de configuración del modelo
│   └── classes.txt           # Lista de clases (101 categorías Food-101)
│
├── app.py                    # Servidor FastAPI principal
├── clients.py                # Cliente Open Food Facts + caché nutricional
├── utils.py                  # Funciones de carga de modelo e inferencia
├── train.py                  # Script de entrenamiento del modelo
├── evaluate_model.py         # Evaluación y métricas del modelo
├── training_config.py        # Configuración de hiperparámetros
├── interface.py              # Interfaz Streamlit (alternativa)
├── quick_test.py             # Pruebas rápidas del modelo
│
├── synonyms.json             # Mapeo de sinónimos para búsqueda nutricional
├── requirements.txt          # Dependencias Python
├── Dockerfile                # Configuración de contenedor Docker
└── README.md                 # Este archivo
```

### Descripción Detallada de Módulos

#### `app.py` - Servidor Backend Principal

**Propósito**: Punto de entrada de la aplicación. Servidor web FastAPI que expone todos los endpoints REST y maneja las peticiones HTTP.

**Funcionalidades clave**:

- **Gestión del ciclo de vida**: Utiliza `lifespan` para cargar el modelo DenseNet201 al iniciar el servidor y liberar recursos al cerrarlo
- **Servicio de archivos estáticos**: Monta la carpeta `Frontend-NutriFood` para servir HTML/CSS/JS
- **Endpoint `/predict`**: Recibe imágenes, ejecuta inferencia del modelo y consulta información nutricional
- **CORS middleware**: Permite peticiones desde cualquier origen para facilitar el desarrollo
- **Integración con Open Food Facts**: Utiliza `NutritionProvider` para enriquecer predicciones con datos nutricionales

**Flujo de ejecución**:

1. Al iniciar: carga modelo + clases + device (CPU/GPU)
2. En cada petición `/predict`: imagen → modelo → top-3 predicciones → búsqueda nutricional → respuesta JSON
3. Maneja errores con códigos HTTP apropiados (400, 404, 500, 503)

#### `utils.py` - Utilidades de Modelo e Inferencia

**Propósito**: Módulo centralizado para manejo del modelo DenseNet201, descarga automática, preprocesamiento de imágenes y predicciones.

**Componentes principales**:

**1. `download_model_if_missing()`**

- Descarga automática del checkpoint desde Google Drive si no existe localmente
- Utiliza `gdown` para obtener el archivo (ID: `10lXt4B9W6ZFBoUALMpCFxoWcRsT8jLWB`)
- Evita re-descargas innecesarias verificando existencia del archivo

**2. `_load_classes()`**

- Lee `meta/classes.txt` con las 101 categorías originales de Food-101
- Retorna solo las primeras 20 clases + "other" (21 clases totales)
- Incluye manejo de errores si el archivo no existe

**3. `_build_model()`**

- Construye la arquitectura exacta: DenseNet201 + clasificador personalizado
- **Arquitectura**:
  - Backbone: DenseNet201 (feature extractor)
  - Clasificador intermedio: 1920 → 1024 (LeakyReLU) → 101
  - Head final: 101 → 21 (clases entrenadas)
- Carga pesos del checkpoint con `map_location='cpu'` para compatibilidad
- Limpia claves del state_dict (remueve prefijo "module.")

**4. `preprocess_image()`**

- Pipeline de transformaciones estándar de ImageNet:
  - Resize a 255px
  - CenterCrop a 224x224
  - Conversión a tensor normalizado
  - Normalización con media/std de ImageNet
- Convierte bytes → PIL Image → Tensor

**5. `predict_from_bytes()`**

- Ejecuta inferencia sobre imagen en formato bytes
- Aplica softmax para obtener probabilidades
- Retorna top-k predicciones con labels y confianza

#### `clients.py` - Integración con Open Food Facts

**Propósito**: Cliente HTTP para consultar la API de Open Food Facts y obtener información nutricional. Incluye sistema de caché y manejo de sinónimos.

**Clases principales**:

**1. `OpenFoodFactsClient`**

- **Método `search()`**: Busca productos en Open Food Facts con términos de búsqueda
- **Método `get_best_nutriments()`**: Obtiene el mejor resultado nutricional para una query
  - Primero busca en caché local (`nutrition_cache.json`)
  - Si no existe, consulta API externa
  - Guarda resultado en caché para futuras consultas
- **Sistema de sinónimos**: Mapea variantes lingüísticas (ej: "ceviche" → "seviche" → "cebiche")
- **Manejo de errores**: Captura excepciones de red y timeouts

**2. `NutritionProvider`**

- Capa de abstracción sobre `OpenFoodFactsClient`
- **Método `get_nutrition_for_labels()`**: Estrategia inteligente para múltiples labels
  - Recibe lista de predicciones (ej: ["pizza", "bruschetta", "cheese_plate"])
  - Genera candidatos expandidos con sinónimos
  - Itera hasta encontrar el primer resultado válido
  - Retorna información estructurada con proveedor + resultado

**Optimizaciones**:

- Cache persistente en JSON para reducir llamadas API
- Normalización de queries (lowercase, sin caracteres especiales)
- Timeout configurable (5 segundos default)

#### `train.py` - Entrenamiento del Modelo

**Propósito**: Script completo para entrenar/fine-tunear el modelo DenseNet201 sobre Food-101.

**Funciones clave**:

**1. `prep_df()`**

- Lee archivos de texto del dataset (`train.txt`, `test.txt`)
- Construye DataFrame con columnas: label, file, path
- Valida existencia de imágenes y filtra archivos faltantes
- Formato esperado: `<clase>/<id_imagen>` (ej: "pizza/123456")

**2. `Food21` (Dataset class)**

- Dataset personalizado de PyTorch
- Carga imágenes desde disco y aplica transformaciones
- Mapea labels a índices numéricos (0-20)
- Convierte imágenes a RGB si es necesario

**3. `build_model()`**

- Similar a `_build_model()` en utils.py pero permite configurar pre-entrenamiento
- Inicialmente congela backbone para entrenar solo clasificador
- Opción `pretrained_backbone` carga pesos de ImageNet

**4. `train()`** - Bucle principal de entrenamiento

- **Preparación de datos**:
  - Carga y filtra dataset según `--train-subset`
  - Aplica augmentations (RandomCrop, Flip) para entrenamiento
  - Soporta WeightedRandomSampler para balancear clases desbalanceadas
- **Configuración de optimización**:

  - Optimizer: AdamW con weight decay
  - Loss: CrossEntropyLoss (con opcional class weights)
  - Scheduler: StepLR o CosineAnnealing
  - Mixed Precision: Utiliza GradScaler en GPU para acelerar

- **Estrategias de fine-tuning** (`--unfreeze`):

  - `none`: Solo head final (default para iteración rápida)
  - `classifier`: Head + clasificador intermedio
  - `head`: Solo head final
  - `all`: Todo el modelo (más lento, mejor precisión)

- **Early stopping**: Detiene si no hay mejora en `--patience` epochs
- **Checkpointing**: Guarda mejor modelo con metadatos (epoch, accuracy, optimizer state)
- **Resuming**: Opción `--resume` para continuar entrenamiento interrumpido

**Argumentos CLI importantes**:

```bash
--epochs: Número de epochs
--batch-size: Tamaño de batch (256 recomendado para GPU)
--lr: Learning rate (1e-3 default)
--train-subset: Porción del dataset (0.3 = 30% para experimentos rápidos)
--pretrained-backbone: Usar pesos de ImageNet
--use-sampler: Balanceo de clases con WeightedRandomSampler
--scheduler: Tipo de scheduler (step, cosine, none)
```

#### `evaluate_model.py` - Evaluación y Métricas

**Propósito**: Script para evaluar el modelo entrenado sobre el conjunto de test y generar métricas detalladas.

**Funciones principales**:

**1. `prep_test_df()`**

- Carga datos de test desde `food-101/meta/test.txt`
- Mapea labels: primeras 20 clases se mantienen, resto → "other"
- Construye DataFrame con paths completos a imágenes

**2. `Food21Dataset`**

- Dataset de evaluación (sin augmentations)
- Solo aplica transformaciones básicas (Resize, CenterCrop, Normalize)

**3. `evaluate()`** - Proceso de evaluación completo

- **Métricas calculadas**:

  - Top-1 Accuracy: Predicción correcta en primer lugar
  - Top-3 Accuracy: Label verdadero dentro del top-3
  - Classification Report: Precision, Recall, F1-Score por clase
  - Confusion Matrix: Visualización de confusiones entre clases

- **Salidas generadas**:
  - Imprime métricas en consola
  - Guarda matriz de confusión visual en `confusion_matrix.png`
  - Reporte detallado por clase usando sklearn

**Uso típico**:

```bash
python evaluate_model.py --checkpoint ./ckpt_finetuned.pt
```

#### `training_config.py` - Configuración de Hiperparámetros

**Propósito**: Archivo centralizado de configuración para experimentación rápida vs producción.

**Parámetros configurables**:

```python
# Dataset sampling
TRAIN_SUBSET_RATIO = 0.3  # 30% para iteración rápida
TEST_SUBSET_RATIO = 0.5   # 50% para validación más rápida

# Training
BATCH_SIZE = 256          # Mayor batch = mejor uso de GPU
NUM_EPOCHS = 10           # Reducido para experimentos
LEARNING_RATE = 1e-3

# Early stopping
TARGET_ACCURACY = 0.85    # Detener al alcanzar este accuracy
PATIENCE_EPOCHS = 3       # Esperar 3 epochs sin mejora
MIN_IMPROVEMENT = 0.005   # Mejora mínima para considerar progreso

# Performance
PIN_MEMORY = True         # Acelera transferencia CPU→GPU
NON_BLOCKING = True       # Transfer asíncrono
```

**Uso recomendado**:

- Desarrollo/debugging: Valores default (subsets pequeños)
- Producción: Cambiar a TRAIN_SUBSET_RATIO=1.0, NUM_EPOCHS=35

#### `interface.py` - Interfaz Streamlit (Alternativa)

**Propósito**: Interfaz gráfica alternativa usando Streamlit para usuarios que prefieren una aplicación de escritorio.

**Funcionalidades**:

- Carga de imágenes mediante file uploader
- Preview de imagen subida
- Slider para ajustar umbral de confianza
- Visualización de predicción principal con métricas tipo dashboard
- Display de información nutricional en columnas (calorías, carbohidratos, proteínas, grasas)
- Expandible con detalles nutricionales completos en JSON

**Ventajas**:

- Más rápido de prototipar que HTML/CSS/JS
- Ideal para demos internas o usuarios técnicos
- No requiere conocimientos de frontend

**Desventaja**:

- Menos personalizable visualmente que la interfaz web
- Requiere ejecutar servidor local (`streamlit run interface.py`)

#### Archivos de Configuración

**`synonyms.json`**

- Mapeo de variantes lingüísticas para mejorar búsqueda nutricional
- Estructura: `{"label_principal": ["sinónimo1", "sinónimo2", ...]}`
- Ejemplos:
  - `"ceviche": ["ceviche", "seviche", "cebiche"]`
  - `"apple_pie": ["apple pie", "tarta de manzana", "pie"]`
- Utilizado por `OpenFoodFactsClient` para expandir búsquedas

**`requirements.txt`**

- Lista todas las dependencias Python del proyecto
- Incluye versiones específicas para reproducibilidad
- Utiliza PyTorch CPU por default (cambiar a GPU si es necesario)
- Instalación: `pip install -r requirements.txt`

**`Dockerfile`**

- Receta para containerizar la aplicación
- Basado en `python:3.10-slim` para menor tamaño
- Pasos:
  1. Instala dependencias del sistema (curl)
  2. Actualiza pip
  3. Instala dependencias Python
  4. Copia código fuente
  5. Expone puerto 8000
  6. Ejecuta servidor uvicorn
- Build: `docker build -t clasificador-comida .`
- Run: `docker run -p 8000:8000 clasificador-comida`

**`meta/classes.txt`**

- Lista de las 101 categorías originales de Food-101
- Una clase por línea (ej: "apple_pie", "baby_back_ribs", etc.)
- El sistema utiliza solo las primeras 20 + "other" para clasificación
- Generado automáticamente al descargar el dataset Food-101

---

## API Endpoints

### Endpoints Principales

| Método | Ruta                         | Descripción                         |
| ------ | ---------------------------- | ----------------------------------- |
| `GET`  | `/`                          | Interfaz web principal              |
| `GET`  | `/health`                    | Estado del servidor y modelo        |
| `POST` | `/predict`                   | Clasificación de imagen + nutrición |
| `GET`  | `/classes`                   | Lista de clases soportadas          |
| `GET`  | `/nutrition?q={food}`        | Consulta nutricional directa        |
| `GET`  | `/off-search?q={query}`      | Búsqueda en Open Food Facts         |
| `GET`  | `/off-suggestions?q={query}` | Sugerencias de productos            |

### Ejemplo de Uso: `/predict`

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@imagen_pizza.jpg"
```

**Response:**

```json
{
  "predictions": [
    {
      "label": "pizza",
      "confidence": 0.9234
    },
    {
      "label": "bruschetta",
      "confidence": 0.0512
    },
    {
      "label": "cheese_plate",
      "confidence": 0.0234
    }
  ],
  "nutrition": {
    "provider": "openfoodfacts",
    "query": "pizza",
    "result": {
      "product_name": "Pizza Margherita",
      "brands": "Various",
      "nutriments": {
        "energy-kcal_100g": 266,
        "carbohydrates_100g": 33.0,
        "proteins_100g": 11.0,
        "fat_100g": 10.0
      },
      "serving_size": "100g"
    }
  }
}
```

---

## Entrenamiento del Modelo

### Dataset Food-101

El proyecto utiliza el dataset [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), que contiene:

- 101 categorías de alimentos
- 101,000 imágenes (1,000 por clase)
- Split: 75,750 entrenamiento / 25,250 prueba

### Proceso de Fine-tuning

1. **Preparar el dataset**

   ```bash
   # Descargar y extraer Food-101
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar xzf food-101.tar.gz
   ```

2. **Configurar parámetros** (editar `training_config.py`)

   ```python
   TRAIN_SUBSET_RATIO = 0.3  # Usa 30% para iteración rápida
   BATCH_SIZE = 256
   NUM_EPOCHS = 10
   ```

3. **Ejecutar entrenamiento**

   ```bash
   python train.py \
     --epochs 10 \
     --batch-size 256 \
     --lr 1e-3 \
     --unfreeze classifier \
     --pretrained-backbone \
     --output ./ckpt_finetuned.pt
   ```

4. **Evaluar el modelo**
   ```bash
   python evaluate_model.py --checkpoint ./ckpt_finetuned.pt
   ```

### Hiperparámetros Recomendados

**Para experimentación rápida:**

```bash
python train.py --train-subset 0.3 --epochs 8 --batch-size 256
```

**Para producción (máxima precisión):**

```bash
python train.py --train-subset 1.0 --epochs 35 --batch-size 128 \
  --unfreeze all --use-sampler --scheduler cosine
```

---

## Interfaz de Usuario

### Funcionalidades del Frontend

1. **Carga de Imágenes**

   - Drag & drop o selección de archivo
   - Vista previa instantánea
   - Formatos soportados: JPG, PNG, JPEG

2. **Análisis Visual**

   - Barra de confianza animada
   - Top-3 predicciones con probabilidades
   - Información nutricional destacada

3. **Chat Nutricional**
   - Consultas en lenguaje natural
   - Respuestas contextuales sobre el alimento detectado
   - Historial de conversación

### Capturas de Pantalla

_(Aquí puedes agregar screenshots de tu interfaz)_

---

## Testing

### Prueba Rápida del Modelo

```bash
python quick_test.py --image ./test_images/burger.jpg
```

### Tests de la API

```bash
# Health check
curl http://localhost:8000/health

# Obtener clases
curl http://localhost:8000/classes

# Consulta nutricional
curl "http://localhost:8000/nutrition?q=pizza"
```

---

## Rendimiento del Modelo

| Métrica              | Valor        |
| -------------------- | ------------ |
| Top-1 Accuracy       | 85.3%        |
| Top-3 Accuracy       | 95.1%        |
| Tamaño del Modelo    | ~77 MB       |
| Tiempo de Inferencia | ~200ms (CPU) |
| Tiempo de Inferencia | ~50ms (GPU)  |

### Matriz de Confusión

El script `evaluate_model.py` genera automáticamente:

- Reporte de clasificación (precision, recall, f1-score)
- Matriz de confusión visual (`confusion_matrix.png`)
- Métricas por clase

---

## Open Food Facts Integration

El sistema integra la API pública de [Open Food Facts](https://world.openfoodfacts.org/) para obtener información nutricional:

- **Base de datos**: 2M+ productos colaborativos
- **Cobertura**: Global (múltiples idiomas)
- **Información**: Calorías, macros, micronutrientes, alérgenos
- **Cache local**: Archivo `nutrition_cache.json` para reducir latencia

### Sistema de Sinónimos

El archivo `synonyms.json` mapea variantes lingüísticas:

```json
{
  "apple_pie": ["apple pie", "tarta de manzana", "pie"],
  "ceviche": ["ceviche", "seviche", "cebiche"]
}
```

---

## Despliegue en Producción

### Docker Compose (Recomendado)

Crear `docker-compose.yml`:

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=./ckpt_finetuned.pt
    restart: unless-stopped
```

Ejecutar:

```bash
docker-compose up -d
```

### Variables de Entorno

```bash
export MODEL_PATH=./ckpt_finetuned.pt
export DRIVE_ID=10lXt4B9W6ZFBoUALMpCFxoWcRsT8jLWB
```

---

## Tecnologías Utilizadas

### Backend

- **FastAPI** 0.100+: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **PyTorch** 2.0+: Deep Learning framework
- **TorchVision**: Modelos y transformaciones de imagen
- **Pillow**: Procesamiento de imágenes
- **Requests**: Cliente HTTP para APIs externas
- **gdown**: Descarga automática desde Google Drive

### Frontend

- **HTML5 + CSS3**: Estructura y estilos responsivos
- **JavaScript Vanilla**: Lógica sin dependencias externas
- **Font Awesome**: Iconografía moderna

### Infraestructura

- **Docker**: Containerización
- **Python 3.10**: Lenguaje base
- **Git**: Control de versiones

---

## Roadmap y Mejoras Futuras

- [ ] Aumentar a 101 clases completas de Food-101
- [ ] Implementar detección de múltiples objetos (YOLO)
- [ ] Agregar estimación de porciones por visión computacional
- [ ] Integrar más fuentes nutricionales (USDA, FatSecret)
- [ ] Aplicación móvil (React Native)
- [ ] Modo offline con modelo cuantizado
- [ ] Soporte multiidioma (i18n)
- [ ] Sistema de usuarios y tracking de comidas

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## Autores

- **Equipo UPAO - 6to Ciclo IA** - [GitHub](https://github.com/gportal-21/NUTRITIONAL-FOOD)

---

## Agradecimientos

- Dataset [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) por ETH Zurich
- [Open Food Facts](https://world.openfoodfacts.org/) por su API abierta
- Comunidad PyTorch por las herramientas de Deep Learning
- FastAPI por su excelente framework

---

## Contacto y Soporte

- **Repositorio**: [NUTRITIONAL-FOOD](https://github.com/gportal-21/NUTRITIONAL-FOOD)
- **Issues**: [GitHub Issues](https://github.com/gportal-21/NUTRITIONAL-FOOD/issues)

---

**Desarrollado por el Equipo de IA - UPAO**
