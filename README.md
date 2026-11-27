# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.

Variables agregadas por el grupo (declaradas en `src/lib/config.py` y en los dos `.env.*.example`):

| Variable | Valor | Para qué |
|---|---|---|
| `BASELINE_EMBEDDING_DIM`, `RESNET18_EMBEDDING_DIM`, `CNN_CUSTOM_EMBEDDING_DIM` | 2048, 512, 256 | Dimensión de cada modelo: cada uno tiene su propia tabla `vector(dim)`. Al arrancar, el backend verifica que `EMBEDDING_DIM` coincida con la de `EMBEDDING_MODEL` |
| `YOLO_IOU_THRESHOLD` | 0.7 | IoU del NMS de YOLO (valor por defecto de ultralytics) |
| `CLASSIFIER_MIN_CONFIDENCE` | 0.30 | Probabilidad mínima para aceptar la raza de un recorte; por debajo, `"unknown"`. Calibrada sobre `valid` |

Valores elegidos: `SIMILARITY_METRIC=cosine`, `SIMILARITY_THRESHOLD=0.8029`, `TOP_K=10` (ver [Resultados](#resultados)).
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

```text
imagen (BGR) ──► preprocesamiento compartido ──► Etapa 1: embedding ──► top-K vecinos en pgvector ──► raza por voto
                 (lib/vision/preprocessing.py)    (ResNet50 o modelo      (tabla por modelo)              o "unknown"
                                                   entrenado)
imagen ──► Etapa 2: clasificador (Modelo A o B) ──► softmax ──► raza o "unknown" (CLASSIFIER_MIN_CONFIDENCE)
imagen ──► Etapa 3: YOLOv8n (perros, imgsz adaptativo) ──► recorte ──► clasificador de la Etapa 2 ──► cajas y razas
```

- **Preprocesamiento único** (`src/lib/vision/preprocessing.py`): BGR a RGB, resize que conserva la relación de aspecto, normalización de ImageNet y padding hasta 224 x 224. Lo usan las notebooks, el entrenamiento y los tres servicios, así lo que se mide es lo que hace la app.
- **Etapa 1** (`similarity_service.py`): `extract_embedding` usa ResNet50 `IMAGENET1K_V1` sin la capa `fc` (2048 dimensiones, normalizado en L2), cargada una sola vez. `search_similar_images` resuelve el top-K en PostgreSQL con el operador `<=>` (o recorre el JSON). `predict_breed_from_neighbors` vota la raza ponderando por similitud y devuelve `"unknown"` si el mejor vecino queda bajo `SIMILARITY_THRESHOLD`.
- **Etapa 2** (`classifier_service.py`): carga el checkpoint del modelo elegido (A `resnet18_finetuned` o B `cnn_custom`), aplica el preprocesamiento compartido y toma la raza de mayor probabilidad del softmax; si esa probabilidad queda bajo `CLASSIFIER_MIN_CONFIDENCE` responde `"unknown"`. `train_classifier` entrena con la receta común (`lib/training/trainer.py`) y `evaluate_classifier` calcula las métricas sobre un split. `extract_custom_embedding` expone la penúltima capa para usar el modelo en la Etapa 1.
- **Etapa 3** (`detection_service.py`): YOLOv8n preentrenado en COCO, filtrado a la clase perro (16), con confianza mínima 0,25 e IoU de NMS 0,7. El tamaño de entrada (`imgsz`) se adapta a cada imagen: el lado mayor redondeado a múltiplo de 32 y acotado entre 320 y 640. Cada caja se recorta y se clasifica con el modelo de la Etapa 2, con el mismo umbral de `"unknown"`.
- **Una base por modelo de embeddings** (mejora opcional): ver [Notas importantes](#notas-importantes).

Código propio en `src/lib/`: `dataset/` (manifiestos, filtrado, splits, augmentation, conjunto independiente), `evaluation/` (métricas de búsqueda y clasificación), `models/`, `training/`, `search/` (regla de voto compartida por la app y las notebooks) y `visualization/`. Tests en `tests/` (`pytest`).

Scripts agregados:

| Script | Para qué |
|---|---|
| `scripts/build_splits.py` | Regenera los splits limpios (`data/splits/splits_v1.csv`) con el mismo código que la notebook 1 |
| `scripts/collect_independent.py` | Junta candidatas de Wikimedia Commons con licencia libre para el conjunto independiente y para las imágenes sin perro que calibran el umbral, y las registra con autor y licencia |
| `scripts/build_index.py --model <modelo>` | Indexa la base del modelo indicado (agregado `--model`) |

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Las dos se entregan ejecutadas y con las salidas en el repositorio.

## Dataset

**70 Dog Breeds-Image Data Set** (Kaggle, CC0): 70 razas, 9.346 imágenes JPG de 224 x 224 en `train`, `valid` y `test`.

**Filtrado** (`1-pre-processing.ipynb`, sección 6): se descartan 748 imágenes. Corruptas y duplicados se buscan en todos los splits; las reglas de calidad se aplican solo a `train`.

| Regla | Descartadas |
|---|---|
| Duplicado dentro de un split (pHash ≤ 8 con coseno ≥ 0,90, o coseno ≥ 0,97 con embeddings del baseline) | 459 |
| Fuga entre splits (la copia se quita de `valid`/`test` y queda en `train`) | 243 |
| Etiqueta contradictoria (la misma foto con dos razas: se descartan todas las copias) | 41 |
| Borrosa (varianza del Laplaciano < 30, solo `train`) | 5 |
| Corrupta, resolución baja o exposición extrema | 0 |

**Splits** (`data/splits/splits_v1.csv`, semilla 42): el 18 % del `test` original eran copias de `train`. Se quitaron las fugas y se completaron `valid` y `test` hasta 10 por raza con imágenes de `train` sin ningún parecido en el dataset. Resultado: `train` 7.198 (56 a 185 por raza), `valid` 700 y `test` 700, sin pares con coseno ≥ 0,95 entre splits.

**Conjunto independiente** (`data/splits/independent_v1.csv`): 134 imágenes que no están en el dataset: 118 de 20 razas (5 a 8 por raza, con los pares que se confunden), 9 escenas con varios perros y 7 sin perros. Fuentes: Wikimedia Commons y Unsplash, con licencia libre y autor registrado por imagen; sin caras visibles. Se controló contra el dataset (se borraron 6 copias). Descarga (solo lectura): [independent_v1.zip](https://drive.google.com/file/d/1u0fhkg3TcoSF4XrwcsO_28gwRSm_FFtG/view?usp=sharing), se descomprime dentro de `data/dataset/`.

**Preprocesamiento y augmentation**: resize con padding (no estira ni recorta los recortes de YOLO). Augmentation online solo en `train` con Albumentations: recorte desde los bordes (simula los recortes de YOLO), flip horizontal, rotación ±12°, brillo y contraste ±20 %, desenfoque y ruido suaves. Sin flips verticales ni rotaciones de 90°.

## Modelos y entrenamiento

- **Baseline de embeddings (Etapa 1)**: ResNet50 preentrenada en ImageNet (`IMAGENET1K_V1`), salida del global average pooling (2048). ImageNet ya incluye unas 118 razas de perro entre sus clases, así que es una línea de base fuerte sin entrenar.
- **Umbral de similitud**: `SIMILARITY_THRESHOLD=0.8029` (`cosine`). Calibrado con el criterio de la clase 4 (barrido, curva ROC y FAR objetivo del 5 %) usando `valid` como consultas conocidas y 92 imágenes sin perro de Wikimedia Commons como desconocidas (`data/splits/calibration_negatives_v1.csv`). Se calibra contra "lo que no es un perro" y no contra razas nuevas, porque una raza que el modelo no vio sigue siendo un perro (con razas apartadas el umbral rechazaba el 45 % de `test`). Lobos y zorros no participan del cálculo: el dataset trata cánidos salvajes como razas (Dingo, Coyote, Dhole, African Wild Dog).
- **Receta común de entrenamiento** (`lib/training/trainer.py`): AdamW, scheduler coseno, label smoothing 0,1, weight decay 1e-4, batch 64, precisión mixta en GPU, early stopping con paciencia 5 sobre la accuracy de `valid` y semilla 42. Se guarda la mejor época según `valid`. Las variantes se eligen con `valid` y `test` se mide una sola vez, con el modelo final.
- **Modelo A** (`resnet18_finetuned`, 11,2 M parámetros): ResNet18 `IMAGENET1K_V1` con la `fc` reemplazada por una capa lineal de 512 a 70. Embedding de 512 (salida del global average pooling). Se compararon tres variantes en `valid`:

  | Variante | Qué se entrena | Accuracy en `valid` |
  |---|---|---|
  | A1 feature extraction | solo la cabeza (LR 1e-3), backbone y BatchNorm congelados | 0,929 |
  | A2 fine-tuning | todo: cabeza 1e-3, backbone 1e-4 | 0,939 |
  | A3 fine-tuning + pesos de clase | igual que A2, con pesos inversos a la frecuencia en la loss | **0,943** |

  Las diferencias están dentro del ruido (un punto de error estándar con 700 imágenes); se eligió A3 por ser la mejor y no costar más. El modelo final es A3 reentrenado (hasta 30 épocas, mejor época 7).
- **Modelo B** (`cnn_custom`, 1,06 M parámetros, desde cero): 5 bloques conv 3x3 + BatchNorm + ReLU + max-pool con 32, 64, 128, 256 y 256 canales, global average pooling, embedding de 256 con ReLU, dropout 0,3 y clasificador de 70. Inicialización de Kaiming, LR 1e-3. Variantes a 30 épocas en `valid`: base 0,557, estadísticas de `train` 0,553, sin BatchNorm 0,544, dropout 0,5 0,477, augmentation fuerte 0,457. Ninguna había convergido y ninguna sobreajustaba, así que más regularización empeoraba. El modelo final es la base con un tope de 60 épocas: 0,716 en `valid` (mejor época 54).
- **YOLO**: YOLOv8n con `imgsz` adaptativo (320 a 640), confianza 0,25 e IoU 0,7. Se comparó con `imgsz` fijo de 320, 640 y 1280 y con YOLO11n (ver [Resultados](#resultados)). YOLO11n detecta algo más en las fotos reales (0,95 contra 0,90, 6 de 118 imágenes) y algo menos en `test`, con latencia parecida; se mantuvo v8n, el modelo por defecto del enunciado, porque la diferencia es chica. `YOLO_CONF_THRESHOLD` se probó con 0,15, 0,25 y 0,40: cambian 1 o 2 imágenes por grupo, así que se mantuvo 0,25.
- **`CLASSIFIER_MIN_CONFIDENCE=0.30`**: calibrado con la curva riesgo-cobertura del Modelo A en `valid`, como el menor umbral cuya accuracy sobre lo aceptado llega a 0,95. En `test` responde el 96,1 % de las imágenes con 94,8 % de acierto y rechaza el 34 % de los errores. Para el Modelo B el mismo criterio da 0,654 y solo respondería el 31 %: el valor de la app está pensado para el A.
- **Checkpoints** (colocarlos en `models/`): [resnet18_finetuned.pth](https://drive.google.com/file/d/103Ze7L58gVNONt5wnHSaHGtY6HdyfTJ0/view?usp=sharing) (43 MB) y [cnn_custom.pth](https://drive.google.com/file/d/1-8W8WxsGniPAzOVRwB6vNj78IYNojocx/view?usp=sharing) (4 MB). Links públicos de solo lectura; la carpeta completa con historiales y experimentos está en [tp2-perros](https://drive.google.com/drive/folders/1yvZ2wy2G7-LhRfiFhQs7YSZCd76mtNQO?usp=sharing).

## Resultados

**Etapa 1** (baseline ResNet50, galería `train`, `cosine`, `TOP_K=10`, sin umbral; IC bootstrap del 95 %):

| Consultas | NDCG@10 | precision@10 | accuracy |
|---|---|---|---|
| `test` (700) | 0,902 [0,887; 0,917] | 0,896 | 0,943 [0,924; 0,960] |
| Conjunto independiente (118) | 0,716 [0,662; 0,772] | 0,707 | 0,797 [0,720; 0,864] |

- `cosine` y `l2` dan los mismos vecinos con embeddings normalizados (cambia solo la escala del score); con vectores crudos `l2` empeora.
- `TOP_K=10` es el mejor valor en `test` y está dentro del ruido del mejor en el independiente.
- Con el umbral elegido, `test` no tiene rechazos; en el independiente se rechazan 4 de 118 fotos reales y 5 de las 7 imágenes sin perro (se aceptan el zorro y el lobo).

**Embeddings de los modelos entrenados en la Etapa 1** (mismo protocolo; silueta coseno de los embeddings de `test`):

| Embeddings | Dimensión | NDCG@10 `test` | accuracy `test` | Silueta | NDCG@10 independiente | accuracy independiente |
|---|---|---|---|---|---|---|
| baseline (ResNet50) | 2048 | 0,902 | 0,943 | 0,297 | **0,716** | **0,797** [0,720; 0,864] |
| Modelo A (ResNet18) | 512 | **0,919** | 0,940 | 0,303 | 0,585 | 0,585 [0,500; 0,669] |
| Modelo B (CNN propia) | 256 | 0,596 | 0,704 | 0,060 | 0,187 | 0,246 |

- En `test` el Modelo A ordena un poco mejor a los vecinos (NDCG +0,017, IC 95 % [0,004; 0,030]), pero la raza votada acierta lo mismo (McNemar 19 contra 21, p = 0,87).
- En fotos reales es al revés y por mucho: el baseline acierta 30 consultas que el A falla y el A solo 5 que el baseline falla (p = 2e-5). El fine-tuning especializó el espacio en el estilo del dataset y pierde fuera de él.
- **Por eso `EMBEDDING_MODEL=baseline`**, que además es el modelo con el que está calibrado `SIMILARITY_THRESHOLD`.

**Etapa 2** (`test`, 700 imágenes, 10 por raza; IC bootstrap del 95 %):

| Modelo | accuracy | IC 95 % | precision | recall | F1 | Parámetros | ms CPU por imagen |
|---|---|---|---|---|---|---|---|
| A (ResNet18 finetune + pesos de clase) | **0,924** | [0,904; 0,943] | 0,933 | 0,924 | **0,923** | 11,2 M | 80,0 (36,2 en ONNX) |
| B (CNN propia) | 0,703 | [0,670; 0,739] | 0,721 | 0,703 | 0,698 | 1,06 M | 39,5 (16,1 en ONNX) |

- La diferencia es clara: 175 imágenes que solo acierta A contra 20 que solo acierta B (McNemar, p = 4e-32). **La app clasifica con el Modelo A.**
- El B tiene 10 veces menos parámetros pero es solo 2 veces más rápido: aplica sus primeras convoluciones a 224 px y hace unas 0,85 GMAC por imagen, contra 1,8 de la ResNet18. Latencias medidas en la CPU de Colab con un hilo de PyTorch (en una CPU local de 4 hilos: 39,8 y 26,6 ms).
- Errores del A: casi todos entre razas parecidas, los mismos pares que el EDA encontró con el baseline (Boston Terrier y Bulldog, Cockapoo y Labradoodle, Lhasa y Shih-Tzu, Pastor Alemán y Malinois). La raza con peor F1 es Bulldog (0,40), y parte de esos errores parecen de etiqueta (Boston Terriers dentro de la carpeta Bulldog). 18 de los 53 errores tienen probabilidad de 0,5 o más: ningún umbral razonable los filtra.
- Errores del B: más repartidos, con confusiones entre razas que no se parecen (Cairn y Yorkie, Basenji y Corgi). Peores razas: Cocker, Chihuahua, Greyhound y Basenji, que el A resuelve con F1 de 0,86 o más.
- Grad-CAM: los dos modelos miran al perro, no al fondo. El A se concentra en la cabeza; el B es más difuso.

**Conjunto independiente** (118 fotos reales de un perro). Si YOLO no encuentra el perro cuenta como error:

| Modelo | Imagen completa | Recorte de YOLO (como la Etapa 3) | F1 con recorte |
|---|---|---|---|
| A | 0,593 | **0,644** [0,559; 0,729] | 0,688 |
| B | 0,195 | 0,356 [0,263; 0,441] | 0,428 |

- **El cambio de dominio es grande**: el A pasa de 92 % en `test` a 64 % en fotos reales. Recortar con YOLO le suma 5 puntos al A (no significativo, McNemar 20 contra 14, p = 0,39) y 16 al B (22 contra 3, p < 0,001). Lo que queda es el dominio en sí: iluminación, resolución, pose y estilo de las fotos.
- Las peores razas son los pares que se confunden, incluidos a propósito: Shih-Tzu (1 de 6), Cockapoo (1 de 5), Labradoodle (2 de 5), Boston Terrier y Bulldog (3 de 6).
- `CLASSIFIER_MIN_CONFIDENCE=0.30` no se transfiere del todo: en los recortes reales la confianza mediana es 0,62, el umbral deja responder el 73 % y lo aceptado acierta el 76 % (72 % sin umbral). No se recalibró con este conjunto para no elegir el umbral sobre los mismos datos en que se informa.

**Etapa 3** (YOLOv8n, confianza 0,25; conteo de perros, porque no hay cajas anotadas):

| `imgsz` | `test` 224 px: detecta | Fotos de un perro: detecta / conteo exacto | Varios perros: conteo exacto | Sin perro: falsas alarmas |
|---|---|---|---|---|
| 320 | **0,85** | 0,85 / 0,83 | 0,44 | 0 de 7 |
| 640 | 0,58 | **0,91** / **0,88** | **0,67** | 1 de 7 |
| 1280 | 0,05 | 0,65 / 0,37 | 0,56 | 1 de 7 |
| **adaptativo** (320 a 640) | **0,85** | 0,90 / **0,88** | **0,67** | 1 de 7 |

- La regla adaptativa queda en el mejor valor de cada dominio. La falsa alarma es un lobo blanco (0,81 de confianza): COCO no tiene la clase lobo y ningún umbral razonable lo filtra.
- Con 9 escenas de varios perros y 7 sin perros, la evidencia es débil.

## Problemas encontrados y soluciones

- **Fugas entre splits**: el 18 % de `test` y el 16 % de `valid` eran copias de `train`, varias recortadas o con otro color (el pHash no las ve: 126 de las 301 fugas solo las detectan los embeddings). Se quitaron y se repusieron `valid` y `test` con imágenes sin parecidos.
- **Etiquetas del dataset**: la misma foto aparece con dos razas (por ejemplo, Rhodesian y Vizsla) y hay Boston Terriers dentro de la carpeta Bulldog; explica parte de los errores de esos pares.
- **Semillas de Albumentations**: `A.Compose(seed=...)` le da la misma semilla a todas las transformaciones, que quedan acopladas (dos flips de p = 0,5 nunca se aplicaban por separado). Se siembra cada transformación por separado; además se re-siembra en cada worker del `DataLoader`, que si no repetía las mismas augmentations.
- **Umbral de "unknown"**: calibrado contra razas nuevas rechazaba casi la mitad de los perros reales; se recalibró contra imágenes sin perro y sin cánidos salvajes (ver [Modelos y entrenamiento](#modelos-y-entrenamiento)).
- **Conjunto independiente**: el dataset de Kaggle incluye fotos de Wikimedia Commons, así que se controló cada imagen contra el dataset (pHash, embeddings y revisión manual desde coseno 0,94); se encontraron y borraron 6 copias. Wikimedia limita las descargas de clientes sin contacto (error 429): el script espera y retoma por grupo.
- **`imgsz` de YOLO con imágenes de 224 px**: con el valor por defecto (640) YOLO agranda la imagen casi tres veces y pierde el perro en 4 de cada 10 imágenes de `test`; con 320 pierde los perros chicos de las fotos grandes. Se resolvió con un `imgsz` adaptativo al tamaño de la imagen, entre 320 y 640 (`adaptive_imgsz` en `detection_service.py`).
- **El Modelo B no llegaba a aprender en 30 épocas**: todas las variantes tenían la mejor época al final y ninguna sobreajustaba, así que más dropout o más augmentation empeoraban. Se reentrenó con un tope de 60 épocas (de 0,557 a 0,716 en `valid`).
- **Sobreajuste del Modelo A**: en el fine-tuning la accuracy de `train` llega a 0,98 mientras `valid` queda en 0,94. El early stopping corta en las épocas 12 a 14 y se guarda la mejor época según `valid`, no la última.
- **Entrenamiento en Colab con cortes de sesión**: cada variante guarda su checkpoint, su historial y un archivo `.done.json` en Drive; al volver a correr la notebook se reutilizan y no se reentrena.
- **Embeddings del modelo ajustado peores en fotos reales**: el Modelo A mejoraba la búsqueda en `test` pero la empeoraba mucho en el independiente (0,585 contra 0,797), así que la Etapa 1 se quedó con el baseline.
- **Confianza del clasificador fuera de dominio**: el umbral calibrado en `valid` da 95 % de acierto en `test` pero 76 % en recortes de fotos reales, porque la red está sobreconfiada en su propio dominio (Guo et al., 2017). Se documentó en lugar de recalibrar con el conjunto de evaluación.
- **Grad-CAM con el backbone congelado**: sin gradiente en el backbone, los hooks no reciben gradientes en la última capa convolucional. Se activa `requires_grad` en la entrada para que el gradiente llegue igual.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo de `EMBEDDING_MODEL`: si no, el backend no arranca y explica por qué.
- Cada modelo de embeddings tiene su propia base: la tabla `embeddings` para el baseline y `embeddings_resnet18_finetuned` / `embeddings_cnn_custom` para los entrenados (o un JSON por modelo con `USE_PGVECTOR=false`). El selector de modelo de la Etapa 1 busca en la base del modelo elegido sin reindexar; cada base se indexa una vez con `python scripts/build_index.py --split train --model <modelo>`. Si se elige un modelo cuya base está vacía, `/status` lo informa con el comando para indexarla. El umbral `SIMILARITY_THRESHOLD` está calibrado para el baseline: con los otros modelos la escala del score cambia.
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
