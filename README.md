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
- **Etapa 2** (`classifier_service.py`): PENDIENTE (Valentín): modelos A y B, entrenamiento y evaluación.
- **Etapa 3** (`detection_service.py`): PENDIENTE (Valentín): YOLOv8n con `imgsz` adaptativo y clasificación de cada recorte.
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
- **Modelo A, Modelo B, hiperparámetros, YOLO y `CLASSIFIER_MIN_CONFIDENCE`**: PENDIENTE (Valentín).
- **Checkpoints**: PENDIENTE: links públicos de `resnet18_finetuned.pth` y `cnn_custom.pth`, a colocar en `models/`.

## Resultados

**Etapa 1** (baseline ResNet50, galería `train`, `cosine`, `TOP_K=10`, sin umbral; IC bootstrap del 95 %):

| Consultas | NDCG@10 | precision@10 | accuracy |
|---|---|---|---|
| `test` (700) | 0,902 [0,887; 0,917] | 0,896 | 0,943 [0,924; 0,960] |
| Conjunto independiente (118) | 0,716 [0,662; 0,772] | 0,707 | 0,797 [0,720; 0,864] |

- `cosine` y `l2` dan los mismos vecinos con embeddings normalizados (cambia solo la escala del score); con vectores crudos `l2` empeora.
- `TOP_K=10` es el mejor valor en `test` y está dentro del ruido del mejor en el independiente.
- Con el umbral elegido, `test` no tiene rechazos; en el independiente se rechazan 4 de 118 fotos reales y 5 de las 7 imágenes sin perro (se aceptan el zorro y el lobo).
- NDCG@10 con los embeddings de los modelos A y B: PENDIENTE (Valentín).

**Etapa 2 y comparación**: PENDIENTE (Valentín): métricas de ambos modelos sobre `test` y el independiente, matriz de confusión, comparación y análisis de errores.

## Problemas encontrados y soluciones

- **Fugas entre splits**: el 18 % de `test` y el 16 % de `valid` eran copias de `train`, varias recortadas o con otro color (el pHash no las ve: 126 de las 301 fugas solo las detectan los embeddings). Se quitaron y se repusieron `valid` y `test` con imágenes sin parecidos.
- **Etiquetas del dataset**: la misma foto aparece con dos razas (por ejemplo, Rhodesian y Vizsla) y hay Boston Terriers dentro de la carpeta Bulldog; explica parte de los errores de esos pares.
- **Semillas de Albumentations**: `A.Compose(seed=...)` le da la misma semilla a todas las transformaciones, que quedan acopladas (dos flips de p = 0,5 nunca se aplicaban por separado). Se siembra cada transformación por separado; además se re-siembra en cada worker del `DataLoader`, que si no repetía las mismas augmentations.
- **Umbral de "unknown"**: calibrado contra razas nuevas rechazaba casi la mitad de los perros reales; se recalibró contra imágenes sin perro y sin cánidos salvajes (ver [Modelos y entrenamiento](#modelos-y-entrenamiento)).
- **Conjunto independiente**: el dataset de Kaggle incluye fotos de Wikimedia Commons, así que se controló cada imagen contra el dataset (pHash, embeddings y revisión manual desde coseno 0,94); se encontraron y borraron 6 copias. Wikimedia limita las descargas de clientes sin contacto (error 429): el script espera y retoma por grupo.
- PENDIENTE (Valentín): problemas de entrenamiento y detección (por ejemplo, el `imgsz` de YOLO con imágenes de 224 px).

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
