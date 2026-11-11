"""Experimento reproducible para la Actividad 04."""
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import load_iris, make_circles, make_moons
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score, v_measure_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
OUTPUT = Path("resultados")
OUTPUT.mkdir(exist_ok=True)
COLORS = ["#007F86", "#B74B24", "#5148A0", "#549229", "#B72864"]


def kmeans(k):
    return KMeans(n_clusters=k, n_init=20, random_state=SEED)


def evaluate(name, model, x, truth, labels):
    assigned = labels != -1
    groups = len(np.unique(labels[assigned]))
    valid = 2 <= groups < assigned.sum()
    return {
        "dataset": name, "modelo": model, "grupos": groups,
        "ruido": int((~assigned).sum()), "cobertura": float(assigned.mean()),
        "ARI": adjusted_rand_score(truth, labels),
        "V-measure": v_measure_score(truth, labels),
        "Silhouette": silhouette_score(x[assigned], labels[assigned]) if valid else None,
        "Davies-Bouldin": davies_bouldin_score(x[assigned], labels[assigned]) if valid else None,
    }


def plot_case(name, x, partitions, ks, inertias):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharex=True, sharey=True,
                             layout="constrained")
    columns = (2, 3) if name == "Iris" else (0, 1)
    for ax, (title, labels) in zip(axes, partitions.items()):
        for label in np.unique(labels):
            points = x[labels == label]
            noise = label == -1
            ax.scatter(points[:, columns[0]], points[:, columns[1]],
                       s=12, marker="x" if noise else "o",
                       color="#53616B" if noise else COLORS[label % len(COLORS)],
                       alpha=.80)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(alpha=.18)
    fig.supxlabel("Longitud del pétalo (z)" if name == "Iris" else "Coordenada X")
    fig.supylabel("Ancho del pétalo (z)" if name == "Iris" else "Coordenada Y")
    fig.savefig(OUTPUT / f"{name}_grupos.png", dpi=175)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 2.5), layout="constrained")
    ax.plot(ks, inertias, "o-", color="#007F86")
    ax.set(xlabel="Número de grupos (k)", ylabel="Inercia", xticks=ks)
    ax.grid(alpha=.23)
    fig.savefig(OUTPUT / f"{name}_codo.png", dpi=175)
    plt.close(fig)


def write_csv(filename, rows):
    with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)




# Mismas configuraciones del proyecto recibido
iris, iris_labels = load_iris(return_X_y=True)
moons, moons_labels = make_moons(n_samples=500, noise=.045, random_state=SEED)
circles, circle_labels = make_circles(n_samples=500, factor=.45, noise=.035, random_state=SEED)
cases = [
    ("Iris", StandardScaler().fit_transform(iris), iris_labels, 3, .5, 5, 12),
    ("Moons", moons, moons_labels, 2, .16, 5, 12),
    ("Circle", circles, circle_labels, 2, .15, 5, 12),
]
metrics, elbows = [], []
ks = list(range(1, 11))
# Limitar hilos también hace más ágil este ejemplo pequeño.
threadpool_limits(limits=1)
print('Datos preparados: Iris 150×4; Moons y Circle 500×2. Semilla 42.')


import warnings
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
predictions, notices, graph_info = {}, [], []
def run_case(case):
    name,x,truth,k,eps,min_samples,neighbors = case
    models = {'K-means': kmeans(k), 'DBSCAN': DBSCAN(eps=eps,min_samples=min_samples),
              'Spectral': SpectralClustering(n_clusters=k,affinity='nearest_neighbors',
                    n_neighbors=neighbors,assign_labels='kmeans',n_init=20,random_state=SEED)}
    partitions={'Referencia':truth}
    for model_name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            labels=model.fit_predict(x)
        notices.extend({'dataset':name,'modelo':model_name,'aviso':str(w.message)} for w in caught)
        partitions[model_name]=labels
        predictions[(name,model_name)]=labels
        metrics.append(evaluate(name,model_name,x,truth,labels))
        pd.crosstab(pd.Series(truth,name='clase_real'),pd.Series(labels,name='cluster')).to_csv(OUTPUT/f'{name}_{model_name}_contingencia.csv')
    A=kneighbors_graph(x,n_neighbors=neighbors,include_self=True)
    graph_info.append({'dataset':name,'componentes':connected_components(A.maximum(A.T),directed=False)[0]})
    inertias=[kmeans(candidate).fit(x).inertia_ for candidate in ks]
    elbows.extend({'dataset':name,'k':candidate,'inercia':inertia} for candidate,inertia in zip(ks,inertias))
    plot_case(name,x,partitions,ks,inertias)
    print(pd.DataFrame([r for r in metrics if r['dataset']==name]).round(4).to_string(index=False))
    for filename in [f'{name}_grupos.png',f'{name}_codo.png']:
        fig,ax=plt.subplots(figsize=(12,3.5) if 'grupos' in filename else (7,3.5))
        ax.imshow(plt.imread(OUTPUT/filename)); ax.axis('off'); fig.tight_layout(); plt.show(); plt.close(fig)


run_case(cases[0])

run_case(cases[1])

run_case(cases[2])

# Análisis adicional: no sustituye las nueve configuraciones base.
from sklearn.neighbors import NearestNeighbors
sensitivity=[]
for name,x,truth,k,eps,min_samples,neighbors in cases:
    for radius in {'Iris':[.3,.5,1.2],'Moons':[.06,.16,.6],'Circle':[.06,.15,.6]}[name]:
        labels=DBSCAN(eps=radius,min_samples=5).fit_predict(x)
        sensitivity.append({'parametro':f'eps={radius}',**evaluate(name,'DBSCAN',x,truth,labels)})
    for n in [3,12,40]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            labels=SpectralClustering(n_clusters=k,affinity='nearest_neighbors',n_neighbors=n,
                     assign_labels='kmeans',n_init=20,random_state=SEED).fit_predict(x)
        notices.extend({'dataset':name,'modelo':f'Spectral vecinos={n}','aviso':str(w.message)} for w in caught)
        sensitivity.append({'parametro':f'n_neighbors={n}',**evaluate(name,'Spectral',x,truth,labels)})
pd.DataFrame(sensitivity).to_csv(OUTPUT/'sensibilidad.csv',index=False)
print(pd.DataFrame(sensitivity).round(4).to_string(index=False))
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,(name,x,truth,k,eps,min_samples,neighbors) in zip(axes,cases):
    distances,_=NearestNeighbors(n_neighbors=5).fit(x).kneighbors(x)
    ax.plot(np.sort(distances[:,-1]),color='#007F86')
    ax.axhline(eps,ls='--',color='#B74B24',label=f'eps={eps}')
    ax.set_title(name);ax.set_xlabel('Observaciones ordenadas');ax.set_ylabel('Distancia al vecino 5');ax.legend()
fig.tight_layout();fig.savefig(OUTPUT/'distancias_vecinos.png',dpi=160);plt.show();plt.close(fig)


import json,platform,sklearn,scipy,matplotlib
DATA=Path('datos');DATA.mkdir(exist_ok=True)
for name,raw,truth in [('Iris',iris,iris_labels),('Moons',moons,moons_labels),('Circle',circles,circle_labels)]:
    frame=pd.DataFrame(raw,columns=load_iris().feature_names if name=='Iris' else ['x1','x2'])
    frame['clase_real']=truth;frame.to_csv(DATA/f'{name}.csv',index=False)
write_csv('metricas.csv',metrics);write_csv('codo.csv',elbows)
pd.DataFrame(graph_info).to_csv(OUTPUT/'grafos.csv',index=False)
(OUTPUT/'advertencias.json').write_text(json.dumps(notices,indent=2,ensure_ascii=False),encoding='utf-8')
versions={'python':platform.python_version(),'scikit-learn':sklearn.__version__,'numpy':np.__version__,
          'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}
(OUTPUT/'versiones.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
assert len(metrics)==9
assert all(0 <= row['cobertura'] <= 1 for row in metrics)
assert evaluate('test','test',moons,moons_labels,np.full(500,-1))['Silhouette'] is None
assert evaluate('test','test',moons,moons_labels,np.zeros(500,dtype=int))['Davies-Bouldin'] is None
print('Verificación correcta: nueve casos y manejo de todo ruido o un único grupo.')
print('Componentes del grafo:',graph_info)
print('Versiones:',versions)


import zipfile
with zipfile.ZipFile('Resultados_Clustering_Nuevo.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in [DATA,OUTPUT]:
        for file in sorted(folder.glob('*')):
            if file.is_file():z.write(file,str(file))
print('Creado Resultados_Clustering_Nuevo.zip')