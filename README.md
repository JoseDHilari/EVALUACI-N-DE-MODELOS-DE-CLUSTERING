# Actividad 04 · Evaluación de modelos de clustering

Universidad Nacional del Altiplano Puno · Ingeniería de Sistemas  
Aprendizaje de Máquina · IX ciclo, grupo B

## Archivos añadidos para trabajar y editar

- **`Clustering_Nuevo_Colab.ipynb`**: notebook autónomo con salidas, explicaciones y nueve experimentos. En Colab, usa Archivo → Subir notebook → Ejecutar todas. No necesitas GPU ni subir datos.
- **`Informe_Clustering_Editable.docx`**: informe en Word con texto y tablas editables. Revisa nombres, docente, fecha y enlace de repositorio antes de entregar. Los gráficos son imágenes regenerables con el código.
- `complementos.py`: reproduce los casos base y añade sensibilidad de parámetros, distancias de vecinos, contingencias, exportación CSV y registro de versiones y advertencias.
- `generar_word.py`: regenera el Word a partir de los resultados. Antes de regenerarlo, conserva una copia si lo editaste manualmente.
- `datos/`: datos originales de los tres casos en CSV.
- `resultados/sensibilidad.csv`, `grafos.csv`, `advertencias.json`, `versiones.json` y tablas `*_contingencia.csv`: evidencias adicionales.

Se conservaron los datos, parámetros y resultados base del proyecto original. El informe se entrega únicamente en Word; `informe.py` se conserva como generador PDF opcional. No se publicó nada en GitHub.

Las celdas del notebook se ejecutaron secuencialmente en Python local y las métricas base se contrastaron con el CSV original. No se ejecutó dentro del servicio Colab. Si cambias parámetros, revisa las interpretaciones escritas y vuelve a ejecutar todas las celdas.

Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python complementos.py
python generar_word.py
```

## Qué contiene el repositorio

- `clustering.py` ajusta K-means, DBSCAN y Spectral Clustering en Iris, Moons y Circle.
- `resultados/` incluye nueve filas de métricas, 30 valores de inercia y seis gráficos.
- `informe.py` convierte esos resultados en el PDF de entrega y permite completar la portada.
- `requirements.txt` declara las dependencias.

## Reproducir el experimento

Con Python 3.12, abre la terminal en esta carpeta y ejecuta:

```bash
python -m pip install -r requirements.txt
python clustering.py
```

Los datos se cargan o generan localmente mediante scikit-learn. Se estandarizan las cuatro variables de Iris y se mantienen las coordenadas originales de los casos sintéticos. La semilla es 42. Los grupos de referencia solo se usan al evaluar. Spectral puede avisar que el grafo de vecinos está desconectado en los casos sintéticos.

| Dataset | Grupos fijados en K-means y Spectral | eps en DBSCAN | min_samples | Vecinos de Spectral |
| --- | ---: | ---: | ---: | ---: |
| Iris | 3 | 0.50 | 5 | 12 |
| Moons | 2 | 0.16 | 5 | 12 |
| Circle | 2 | 0.15 | 5 | 12 |

ARI y V-measure se calculan con todas las muestras. Silhouette y Davies-Bouldin excluyen la etiqueta de ruido (-1). La cobertura permite ver cuántos puntos participaron en las métricas internas. El codo usa la inercia de K-means para k de 1 a 10.

## Entrega en GitHub y aula virtual

1. Crea el repositorio del equipo en GitHub y sube **el contenido** de esta carpeta, incluido `README.md` en la raíz.
2. Comprueba que el docente pueda abrir el repositorio y copia la URL.
3. Genera el PDF final con los nombres del equipo (entre uno y cuatro) y el enlace real:

```bash
python informe.py --github "https://github.com/USUARIO/REPOSITORIO" --integrantes "Ludwing Riva Ticona" "Nombre 2" "Nombre 3" "Nombre 4"
```

4. Abre `Actividad_04_Clustering.pdf`, verifica la portada y el enlace, y entrega ese PDF en el aula virtual.

## Exposición para cuatro integrantes

| Persona | Parte | Qué explicar con la gráfica |
| --- | --- | --- |
| 1 | Objetivo, metodología e Iris | Qué significa agrupar sin etiquetas, por qué se estandarizó Iris y por qué el mejor Silhouette de DBSCAN se calcula sobre menos flores. Citar cobertura 77,3 % y ARI 0,620 de K-means. |
| 2 | Moons | Señalar cómo K-means corta los arcos y por qué DBSCAN y Spectral obtienen ARI 1,000 aun con Silhouette menor. Mostrar que el codo sugiere dos grupos compactos sin garantizar la forma buscada. |
| 3 | Circle | Señalar el corte por sectores y los anillos recuperados. Explicar que Davies-Bouldin sube a 304,339 cuando los centroides de ambos anillos casi coinciden. |
| 4 | Modelos e hiperparámetros | Comparar el hiperplano entre centroides, la expansión por densidad y el grafo de vecinos. Cerrar explicando cuándo sirven las métricas externas e internas. |

Dedicar alrededor de dos minutos a cada parte y usar los paneles de predicciones para señalar las diferencias.
