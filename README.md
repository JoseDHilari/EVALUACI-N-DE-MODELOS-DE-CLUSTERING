# Evaluación de Modelos de Clustering

**Universidad Nacional del Altiplano Puno**  
Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas  
Escuela Profesional de Ingeniería de Sistemas  

* **Curso:** Aprendizaje Máquina (IX Semestre - Grupo B)
* **Docente:** Ing. Mayenka Fernandez Chambi
* **Integrantes:**
  * Hilari Abarca, José Diego
  * Luna Luque, Paolo Boris
  * Quispe Torres, Alex Junior
  * Riva Ticona, Ludwing

---

## Descripción del Proyecto

Este repositorio contiene la implementación, evaluación y análisis comparativo de tres algoritmos de agrupamiento no supervisado (**K-Means**, **DBSCAN** y **Spectral Clustering**) aplicados sobre tres conjuntos de datos con distintas geometrías y distribuciones: **Iris**, **Moons** y **Circle**.

El estudio contrasta el desempeño de fronteras lineales frente a enfoques basados en densidad y teoría de grafos, evaluando los resultados mediante métricas de validación externa (**ARI**, **V-measure**), métricas de validación interna (**Silhouette**, **Davies-Bouldin**), análisis de cobertura frente al ruido e inspección del método del codo (inercia).

---

## Estructura del Repositorio

* **`Clustering_Nuevo_Colab.ipynb`**: Notebook autónomo con el flujo completo de los nueve experimentos principales, visualizaciones, análisis de sensibilidad e interpretación de resultados.
* **`clustering.py`**: Script principal para el ajuste de K-Means, DBSCAN y Spectral Clustering en los tres conjuntos de datos base, cálculo de métricas e inercia, y generación de gráficas comparativas.
* **`complementos.py`**: Script de extensión que reproduce los casos base e incorpora pruebas de sensibilidad de hiperparámetros, cálculo de distancias al $k$-ésimo vecino, tablas de contingencia y registro de entorno.
* **`datos/`**: Conjuntos de datos utilizados en los experimentos en formato CSV (`iris`, `moons` y `circle`).
* **`resultados/`**: Métricas consolidadas, tablas de inercia (para $k$ de 1 a 10), pruebas de sensibilidad (`sensibilidad.csv`), matrices de contingencia (`*_contingencia.csv`), metadatos de ejecución y gráficos generados en formato PNG.
* **`requirements.txt`**: Especificación de librerías y dependencias necesarias para la reproducción del entorno.

---

## Diseño Experimental y Parámetros

Los conjuntos de datos se obtienen y generan a través de `scikit-learn` fijando una semilla aleatoria (`random_state = 42`) para garantizar la reproducibilidad exacta. Las etiquetas reales se reservan exclusivamente para la validación externa y los paneles visuales de referencia.

| Dataset | Observaciones | Preparación y Características | $k$ (K-Means / Spectral) | `eps` (DBSCAN) | `min_samples` (DBSCAN) | `n_neighbors` (Spectral) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Iris** | 150 | 4 variables estandarizadas (`StandardScaler`), 3 especies | 3 | 0.50 | 5 | 12 |
| **Moons** | 500 | Coordenadas originales, 2 clases (`noise=0.045`) | 2 | 0.16 | 5 | 12 |
| **Circle** | 500 | Coordenadas originales, 2 clases (`factor=0.45`, `noise=0.035`) | 2 | 0.15 | 5 | 12 |

* **Configuración adicional:** K-Means utiliza inicialización `k-means++` con `n_init=20`. Spectral Clustering emplea matriz de afinidad por vecinos más cercanos (`affinity='nearest_neighbors'`), asignación mediante `kmeans` y `n_init=20`. DBSCAN utiliza distancia euclidiana estándar.
* **Tratamiento del ruido:** Las métricas externas (ARI y V-measure) evalúan la totalidad de la muestra tratando el ruido (`-1`) como una categoría adicional. Las métricas internas (Silhouette y Davies-Bouldin) se calculan exclusivamente sobre las observaciones asignadas a un clúster válido, reportando el porcentaje de **cobertura** correspondiente.

---

## Resumen de Resultados Principales

| Dataset | Modelo | Grupos | Ruido | Cobertura | ARI | V-measure | Silhouette | Davies-Bouldin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | K-Means | 3 | 0 | 100.0% | **0.620** | **0.659** | 0.460 | 0.834 |
| **Iris** | DBSCAN | 2 | 34 | 77.3% | 0.442 | 0.511 | **0.656** | **0.494** |
| **Iris** | Spectral | 3 | 0 | 100.0% | 0.580 | 0.617 | 0.460 | 0.832 |
| **Moons** | K-Means | 2 | 0 | 100.0% | 0.253 | 0.192 | **0.489** | **0.782** |
| **Moons** | DBSCAN | 2 | 0 | 100.0% | **1.000** | **1.000** | 0.334 | 1.152 |
| **Moons** | Spectral | 2 | 0 | 100.0% | **1.000** | **1.000** | 0.334 | 1.152 |
| **Circle** | K-Means | 2 | 0 | 100.0% | -0.002 | 0.000 | **0.343** | **1.195** |
| **Circle** | DBSCAN | 2 | 0 | 100.0% | **1.000** | **1.000** | 0.136 | 304.339 |
| **Circle** | Spectral | 2 | 0 | 100.0% | **1.000** | **1.000** | 0.136 | 304.339 |

### Hallazgos Clave
1. **Iris:** K-Means logra la mayor concordancia externa (ARI `0.620`). Aunque DBSCAN obtiene los mejores indicadores internos (Silhouette `0.656`), lo hace fusionando dos especies y descartando el 22.7% de las muestras como ruido.
2. **Moons:** Al poseer fronteras no convexas, K-Means secciona ambos arcos mediante un hiperplano lineal (ARI `0.253`), a pesar de reportar una mayor compacidad euclidiana en Silhouette. Tanto DBSCAN como Spectral Clustering recuperan la estructura exacta (ARI `1.000`).
3. **Circle:** Los anillos concéntricos comparten prácticamente el mismo centro geométrico, lo que provoca que K-Means divida los datos en sectores simétricos (ARI `-0.002`) y que el índice Davies-Bouldin se eleve a `304.339` en las soluciones perfectas de DBSCAN y Spectral debido a la cercanía entre los centroides de ambos anillos.

---

## Instrucciones de Ejecución

### 1. Ejecución Local (Python 3.10+)

Clona el repositorio, instala las dependencias y ejecuta los scripts desde la raíz del proyecto:

```bash
git clone [https://github.com/JoseDHilari/EVALUACI-N-DE-MODELOS-DE-CLUSTERING.git](https://github.com/JoseDHilari/EVALUACI-N-DE-MODELOS-DE-CLUSTERING.git)
cd EVALUACI-N-DE-MODELOS-DE-CLUSTERING
python -m pip install -r requirements.txt
