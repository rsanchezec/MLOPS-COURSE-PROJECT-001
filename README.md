# MLOPS-PROJECT-001: Clasificación de Reservas Hoteleras 🏨

Un pipeline integral de MLOps para predecir cancelaciones de reservas hoteleras usando machine learning. Este proyecto implementa un flujo de trabajo ML completo de extremo a extremo con ingesta de datos desde Google Cloud Storage, preprocesamiento, ingeniería de características, entrenamiento de modelos con ajuste de hiperparámetros y seguimiento de experimentos usando MLflow.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Características](#características)
- [Stack Tecnológico](#stack-tecnológico)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Componentes del Pipeline](#componentes-del-pipeline)
- [Rendimiento del Modelo](#rendimiento-del-modelo)
- [Integración con MLflow](#integración-con-mlflow)
- [Contribuir](#contribuir)

## 🎯 Descripción del Proyecto

Este proyecto predice cancelaciones de reservas hoteleras utilizando técnicas de machine learning. El pipeline procesa datos de reservas hoteleras, aplica técnicas avanzadas de preprocesamiento incluyendo SMOTE para manejo de datos desbalanceados, y entrena un clasificador LightGBM con ajuste automático de hiperparámetros.

**Objetivos Clave:**
- Construir un pipeline robusto de MLOps para predicción de reservas hoteleras
- Implementar procesamiento automático de datos e ingeniería de características
- Aplicar técnicas avanzadas de ML para clasificación desbalanceada
- Rastrear experimentos y rendimiento del modelo usando MLflow
- Mantener calidad del código con logging y manejo de excepciones adecuado

## 📁 Estructura del Proyecto

```
MLOPS-PROJECT-001/
├── artifacts/                          # Artefactos generados
│   ├── models/                         # Modelos entrenados
│   │   └── lgbm_model.pkl
│   ├── processed/                      # Datasets procesados
│   │   ├── processed_test.csv
│   │   └── processed_train.csv
│   └── raw/                           # Datasets crudos
│       ├── raw.csv
│       ├── test.csv
│       └── train.csv
├── config/                            # Archivos de configuración
│   ├── __init__.py
│   ├── config.yaml                    # Configuración principal
│   ├── gcloud.json                    # Credenciales GCP
│   ├── model_params.py               # Hiperparámetros del modelo
│   └── paths_config.py               # Configuración de rutas
├── logs/                             # Logs de la aplicación
│   └── log_2025-08-27.log
├── mlruns/                           # Seguimiento de experimentos MLflow
│   ├── 0/                           # Ejecuciones de experimentos
│   └── models/
├── notebook/                         # Notebooks de Jupyter
│   └── notebook.ipynb
├── pipeline/                         # Pipelines de ML
│   ├── __init__.py
│   └── training_pipeline.py         # Pipeline principal de entrenamiento
├── src/                             # Código fuente
│   ├── __init__.py
│   ├── custom_exception.py          # Manejo personalizado de excepciones
│   ├── data_ingestion.py            # Ingesta de datos desde GCP
│   ├── data_preprocessing.py        # Preprocesamiento e ingeniería de características
│   ├── logger.py                    # Configuración de logging
│   └── model_training.py            # Entrenamiento y evaluación del modelo
├── static/                          # Archivos estáticos web
├── templates/                       # Plantillas HTML para la aplicación web
│   └── index.html                  # Página principal de predicciones
├── utils/                           # Funciones utilitarias
│   ├── __init__.py
│   └── common_functions.py          # Funciones utilitarias comunes
├── application.py                   # Aplicación Flask para predicciones web
├── requirements.txt                 # Dependencias de Python
├── setup.py                        # Configuración del paquete
└── README.md                       # Documentación del proyecto
```

## ✨ Características

### 🔄 Pipeline de Datos
- **Ingesta Automática de Datos**: Descarga datos desde Google Cloud Storage
- **División Inteligente de Datos**: División train-test configurable con estratificación
- **Validación de Datos**: Verificaciones exhaustivas de calidad y validación de datos

### 🛠️ Preprocesamiento de Datos
- **Ingeniería Avanzada de Características**: 
  - Codificación por etiquetas para variables categóricas
  - Manejo de asimetría con transformación logarítmica
  - Selección de características usando importancia de Random Forest
- **Manejo de Datos Desbalanceados**: Sobremuestreo SMOTE para datasets balanceados
- **Calidad de Datos**: Eliminación de duplicados y manejo de valores faltantes

### 🤖 Entrenamiento del Modelo
- **Clasificador LightGBM**: Gradient boosting de alto rendimiento
- **Ajuste de Hiperparámetros**: RandomizedSearchCV automatizado
- **Evaluación del Modelo**: Métricas exhaustivas (Precisión, Exactitud, Recall, F1-score)

### 📊 Seguimiento de Experimentos
- **Integración con MLflow**: Seguimiento completo de experimentos
- **Logging de Artefactos**: Almacenamiento de modelos, datasets y métricas
- **Seguimiento de Parámetros**: Logging de hiperparámetros y configuraciones

### 🔍 Monitoreo y Logging
- **Logging Estructurado**: Logging detallado de operaciones con timestamps
- **Manejo de Excepciones**: Clases de excepción personalizadas con seguimiento de errores
- **Monitoreo de Rendimiento**: Seguimiento de métricas de entrenamiento y evaluación

## 🛠️ Stack Tecnológico

- **Machine Learning**: scikit-learn, LightGBM, imbalanced-learn
- **Procesamiento de Datos**: pandas, numpy
- **Integración en la Nube**: google-cloud-storage
- **Seguimiento de Experimentos**: MLflow
- **Configuración**: PyYAML
- **Visualización**: seaborn
- **Framework Web**: Flask (aplicación web para predicciones)
- **Desarrollo**: setuptools

## 🚀 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/rsanchezec/MLOPS-COURSE-PROJECT-001.git
   cd MLOPS-PROJECT-001
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv mlops_env
   source mlops_env/bin/activate  # En Windows: mlops_env\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Instalar el paquete:**
   ```bash
   pip install -e .
   ```

## ⚙️ Configuración

### 1. Configuración de Google Cloud
- Coloca tu archivo JSON de cuenta de servicio GCP en `config/gcloud.json`
- Asegúrate de que la cuenta de servicio tenga acceso a Google Cloud Storage

### 2. Archivo de Configuración (`config/config.yaml`)
```yaml
data_ingestion:
  bucket_name: "nombre_de_tu_bucket_gcp"
  bucket_file_name: "Hotel_Reservations.csv"
  train_ratio: 0.8

data_processing:
  categorical_columns:
    - type_of_meal_plan
    - required_car_parking_space
    - room_type_reserved
    - market_segment_type
    - repeated_guest
    - booking_status
  numerical_columns:
    - no_of_adults
    - no_of_children
    - no_of_weekend_nights
    # ... agregar más según sea necesario
  skewness_threshold: 5
  no_of_features: 10
```

### 3. Parámetros del Modelo (`config/model_params.py`)
Personaliza el espacio de búsqueda de hiperparámetros:
- Distribución de parámetros de LightGBM
- Configuración de RandomizedSearchCV
- Configuración de validación cruzada

## 🎯 Uso

### Ejecutar Pipeline Completo
```bash
python pipeline/training_pipeline.py
```

### Ejecutar Componentes Individuales

**Ingesta de Datos:**
```bash
python src/data_ingestion.py
```

**Preprocesamiento de Datos:**
```bash
python src/data_preprocessing.py
```

**Entrenamiento del Modelo:**
```bash
python src/model_training.py
```

### Aplicación Web de Predicciones
```bash
uv run application.py
# o alternativamente:
python application.py
```
Accede a la aplicación web en `http://localhost:8080` para realizar predicciones interactivas de reservas hoteleras.

### Interfaz MLflow
```bash
mlflow ui
```
Accede a la interfaz MLflow en `http://localhost:5000` para ver experimentos, métricas y artefactos.

## 🔧 Componentes del Pipeline

### 1. Ingesta de Datos (`src/data_ingestion.py`)
- **Integración con GCP**: Descarga datasets desde Google Cloud Storage
- **División de Datos**: División automática train-test con ratios configurables
- **Manejo de Errores**: Manejo robusto de errores para operaciones en la nube

### 2. Preprocesamiento de Datos (`src/data_preprocessing.py`)
- **Ingeniería de Características**: 
  - Codificación categórica con seguimiento de mapeo
  - Corrección de asimetría en características numéricas
  - Selección basada en importancia de características
- **Balanceado de Datos**: Implementación de SMOTE para datasets desbalanceados
- **Aseguramiento de Calidad**: Validación y limpieza de datos

### 3. Entrenamiento del Modelo (`src/model_training.py`)
- **Algoritmo**: LightGBM con ajuste extensivo de hiperparámetros
- **Optimización**: RandomizedSearchCV para búsqueda eficiente de parámetros
- **Evaluación**: Evaluación multi-métrica con logging detallado
- **Persistencia**: Serialización de modelos y gestión de artefactos

### 4. Aplicación Web (`application.py`)
- **Interfaz Flask**: Aplicación web para predicciones en tiempo real
- **Formulario Interactivo**: Interfaz HTML para ingresar datos de reservas
- **Predicciones en Vivo**: Predicciones instantáneas usando el modelo entrenado
- **Despliegue Local**: Servidor web en puerto 8080 para pruebas

### 5. Utilidades (`utils/common_functions.py`)
- **Gestión de Configuración**: Lectura y validación de archivos YAML
- **Carga de Datos**: Carga estandarizada de datos con manejo de errores
- **Logging**: Configuración centralizada de logging

## 📈 Rendimiento del Modelo

El pipeline rastrea múltiples métricas de evaluación:

- **Precisión (Accuracy)**: Precisión general de predicción
- **Exactitud (Precision)**: Tasa de verdaderos positivos para cancelaciones de reservas
- **Recall**: Cobertura de cancelaciones reales
- **F1-Score**: Métrica balanceada precision-recall

Todas las métricas se registran automáticamente en MLflow para seguimiento y comparación de experimentos.

## 📊 Integración con MLflow

### Características del Seguimiento de Experimentos:
- **Logging de Parámetros**: Todos los hiperparámetros del modelo
- **Seguimiento de Métricas**: Métricas de rendimiento a través de ejecuciones
- **Almacenamiento de Artefactos**: Modelos, datasets y visualizaciones
- **Comparación de Ejecuciones**: Comparación lado a lado de experimentos
- **Versionado de Modelos**: Gestión automática de versiones de modelos

### Accediendo a MLflow:
1. Iniciar interfaz MLflow: `mlflow ui`
2. Navegar a `http://localhost:5000`
3. Explorar experimentos, comparar ejecuciones y descargar artefactos

## 🤝 Contribuir

1. Fork el repositorio
2. Crear una rama de característica (`git checkout -b feature/caracteristica-increible`)
3. Commit tus cambios (`git commit -m 'Agregar característica increíble'`)
4. Push a la rama (`git push origin feature/caracteristica-increible`)
5. Abrir un Pull Request

### Guías de Desarrollo:
- Seguir las guías de estilo PEP 8
- Agregar logging exhaustivo para nuevas características
- Incluir manejo de errores para todas las operaciones
- Actualizar documentación para nuevos componentes
- Probar cambios exhaustivamente antes de enviar

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autor

**rsanchez** - *Trabajo inicial*

## 🙏 Agradecimientos

- Proveedores del dataset de reservas hoteleras
- Comunidad MLflow por las excelentes herramientas de seguimiento de experimentos
- Desarrolladores de LightGBM por el framework de gradient boosting de alto rendimiento
- Comunidad scikit-learn por las herramientas integrales de ML

---

*Este proyecto demuestra prácticas modernas de MLOps incluyendo pipelines automatizados, seguimiento de experimentos y flujos de trabajo de machine learning reproducibles.*