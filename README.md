# 🦷 DentalVision EDA

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface" alt="HF Spaces">
  <img src="https://img.shields.io/github/license/ErnestoSCL/dentalvision-eda" alt="License">
</p>

> **Interactive EDA dashboard** for a dental image classification dataset (4 classes: Prosthesis · Healthy · Cavity · Other).  
> Built with **Streamlit + Plotly**, deployable on **Hugging Face Spaces** via Docker.

---

## 📸 Dashboard Preview

| Página | Descripción |
|--------|-------------|
| 🏠 **Inicio** | Métricas globales, distribución en donut, tabla resumen |
| 📊 **Distribución de Clases** | Barras, comparativa Train/Test, test chi-cuadrado |
| 🔬 **Propiedades de Imagen** | KDE de brillo/contraste/RGB/aspect, correlaciones |
| 🔍 **Calidad de Datos** | Duplicados (MD5), data leakage, outliers IQR |
| 🖼️ **Galería** | Grid de imágenes por clase + histograma de píxeles |

---

## 🚀 Deploy en Hugging Face Spaces

1. Crea un nuevo Space en [huggingface.co/spaces](https://huggingface.co/spaces)  
   - SDK: **Docker**  
   - Visibilidad: Public / Private  
2. Conecta este repositorio GitHub al Space  
3. HF construirá y lanzará la app automáticamente en el **puerto 7860**

> ⚠️ Los archivos `.parquet` (~105 MB) se incluyen en la imagen Docker.  
> Si el dataset es demasiado grande, considera cargarlo desde [HF Datasets](https://huggingface.co/docs/datasets).

---

## 🐳 Correr con Docker

```bash
# Build
docker build -t dentalvision-eda .

# Run
docker run -p 7860:7860 dentalvision-eda

# Abrir: http://localhost:7860
```

---

## 💻 Correr localmente (sin Docker)

```bash
# 1. Clonar
git clone https://github.com/ErnestoSCL/dentalvision-eda.git
cd dentalvision-eda

# 2. Entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt

# 4. Lanzar
streamlit run app.py
# → http://localhost:8501
```

---

## 📁 Estructura del Proyecto

```
dentalvision-eda/
├── app.py                          # App principal Streamlit (5 páginas)
├── utils/
│   ├── __init__.py
│   └── data_loader.py              # Carga y caché de datos
├── Dockerfile                      # Para Hugging Face Spaces
├── .dockerignore
├── requirements.txt
├── data_analysis.ipynb             # Notebook EDA detallado
├── train-00000-of-00001.parquet    # Dataset de entrenamiento
└── test-00000-of-00001.parquet     # Dataset de prueba
```

---

## 📊 Dataset

| Propiedad | Valor |
|-----------|-------|
| Clases | 4 (Prótesis, Sano, Caries, Otro) |
| Formato | Parquet con imágenes binarias |
| Split | Train + Test |
| Análisis completo | `data_analysis.ipynb` |

---

## 🛠️ Stack Tecnológico

| Herramienta | Uso |
|-------------|-----|
| **Streamlit** | Framework UI |
| **Plotly** | Gráficos interactivos |
| **Pandas / NumPy** | Manipulación de datos |
| **Pillow** | Procesamiento de imágenes |
| **SciPy** | Estadísticas (KDE, chi-cuadrado) |
| **Docker** | Empaquetado y deploy |

---

## 📄 Licencia

MIT © 2026 [ErnestoSCL](https://github.com/ErnestoSCL)
