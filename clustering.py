"""Experimento reproducible para la Actividad 04."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import load_iris, make_circles, make_moons
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score, v_measure_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
OUTPUT = Path(__file__).resolve().parent / "resultados"
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


def main():
    OUTPUT.mkdir(exist_ok=True)
    iris, iris_labels = load_iris(return_X_y=True)
    moons, moons_labels = make_moons(n_samples=500, noise=.045, random_state=SEED)
    circles, circle_labels = make_circles(n_samples=500, factor=.45, noise=.035,
                                         random_state=SEED)
    cases = [
        ("Iris", StandardScaler().fit_transform(iris), iris_labels, 3, .5, 5, 12),
        ("Moons", moons, moons_labels, 2, .16, 5, 12),
        ("Circle", circles, circle_labels, 2, .15, 5, 12),
    ]
    metrics, elbows = [], []
    ks = list(range(1, 11))
    for name, x, truth, k, eps, min_samples, neighbors in cases:
        models = {
            "K-means": kmeans(k),
            "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
            "Spectral": SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                          n_neighbors=neighbors, assign_labels="kmeans",
                                          n_init=20, random_state=SEED),
        }
        partitions = {"Referencia": truth}
        for model_name, model in models.items():
            labels = model.fit_predict(x)
            partitions[model_name] = labels
            metrics.append(evaluate(name, model_name, x, truth, labels))
        inertias = [kmeans(candidate).fit(x).inertia_ for candidate in ks]
        elbows.extend({"dataset": name, "k": candidate, "inercia": inertia}
                      for candidate, inertia in zip(ks, inertias))
        plot_case(name, x, partitions, ks, inertias)
    write_csv("metricas.csv", metrics)
    write_csv("codo.csv", elbows)
    for row in metrics:
        print(row)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
