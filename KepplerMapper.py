# ============================================================
# Análisis Topológico de Datos - Algoritmo Mapper
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# ============================================================

import numpy as np
import pandas as pd
import kmapper as km
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import json

# ============================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================
data = load_breast_cancer()
X = data.data        # 569 registros, 30 características
y = data.target      # 0 = maligno, 1 = benigno

# Normalización con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2. CÁLCULO DE LA FUNCIÓN DE FILTRO (PCA)
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Usamos las dos primeras componentes principales como lente
lens = X_pca

# ============================================================
# 3. CONSTRUCCIÓN DEL GRAFO CON MAPPER
# ============================================================
mapper = km.KeplerMapper(verbose=1)

# DBSCAN puede generar 0 nodos en este dataset con ciertos parámetros,
# así que aquí usamos KMeans para obtener un grafo estable y visualizable.
graph = mapper.map(
    lens,
    X_scaled,
    clusterer=KMeans(n_clusters=3, random_state=42),
    cover=km.Cover(n_cubes=10, perc_overlap=0.5)
)

# ============================================================
# 4. VISUALIZACIÓN INTERACTIVA → HTML
# ============================================================
# Colorear nodos según promedio del diagnóstico (0=maligno, 1=benigno)
mapper.visualize(
    graph,
    path_html="mapper_output.html",
    title="Mapper - Breast Cancer Wisconsin",
    color_values=y,
    color_function_name="Diagnóstico (0=Maligno, 1=Benigno)"
)
print("Visualización guardada en mapper_output.html")

# ============================================================
# 5. EXPORTACIÓN DEL GRAFO COMO JSON Y CSV
# ============================================================
# Exportar nodos y aristas a JSON
graph_data = {
    "nodes": {nodo: indices.tolist() if hasattr(indices, 'tolist') 
              else indices for nodo, indices in graph["nodes"].items()},
    "edges": graph["links"]
}
with open("mapper_graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)
print("Grafo exportado en mapper_graph.json")

# Exportar aristas a CSV
edges_list = []
for nodo, vecinos in graph["links"].items():
    for vecino in vecinos:
        edges_list.append({"source": nodo, "target": vecino})

edges_df = pd.DataFrame(edges_list)
edges_df.to_csv("mapper_edges.csv", index=False)
print("Aristas exportadas en mapper_edges.csv")

# Exportar nodos a CSV
nodes_list = []
for nodo, indices in graph["nodes"].items():
    nodes_list.append({
        "node_id": nodo,
        "num_points": len(indices),
        "avg_diagnosis": np.mean(y[indices]),
        "points": str(indices)
    })

nodes_df = pd.DataFrame(nodes_list)
nodes_df.to_csv("mapper_nodes.csv", index=False)
print("Nodos exportados en mapper_nodes.csv")