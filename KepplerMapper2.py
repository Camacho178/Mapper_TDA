# ============================================================
# Análisis Topológico de Datos - Algoritmo Mapper
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# Configuración 2: n_cubes=15, perc_overlap=0.3
# ============================================================

import numpy as np
import pandas as pd
import kmapper as km
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import json

# 1. CARGA Y PREPROCESAMIENTO
data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. FUNCIÓN DE FILTRO (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Usamos las dos primeras componentes principales como lente
lens = X_pca

# 3. CONSTRUCCIÓN DEL GRAFO - CONFIGURACIÓN 2
mapper = km.KeplerMapper(verbose=1)

graph = mapper.map(
    lens,
    X_scaled,
    clusterer=KMeans(n_clusters=3, random_state=42),
    cover=km.Cover(n_cubes=15, perc_overlap=0.3)  # <-- parámetros distintos
)

# 4. VISUALIZACIÓN → HTML
mapper.visualize(
    graph,
    path_html="mapper_output2.html",  # <-- archivo distinto
    title="Mapper Config 2 - n_cubes=15, perc_overlap=0.3",
    color_values=y,
    color_function_name="Diagnóstico (0=Maligno, 1=Benigno)"
)
print("Visualización guardada en mapper_output2.html")

# 5. EXPORTACIÓN
graph_data = {
    "nodes": {nodo: indices.tolist() if hasattr(indices, 'tolist') 
              else indices for nodo, indices in graph["nodes"].items()},
    "edges": graph["links"]
}
with open("mapper_graph2.json", "w") as f:
    json.dump(graph_data, f, indent=2)

edges_list = []
for nodo, vecinos in graph["links"].items():
    for vecino in vecinos:
        edges_list.append({"source": nodo, "target": vecino})

pd.DataFrame(edges_list).to_csv("mapper_edges2.csv", index=False)

nodes_list = []
for nodo, indices in graph["nodes"].items():
    nodes_list.append({
        "node_id": nodo,
        "num_points": len(indices),
        "avg_diagnosis": np.mean(y[indices]),
        "points": str(indices)
    })

pd.DataFrame(nodes_list).to_csv("mapper_nodes2.csv", index=False)
print("Archivos exportados correctamente")