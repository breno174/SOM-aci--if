import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from implement.som import SOM
from implement.kmenasSimple import KMeans as SimpleKMeans
from src.helper.plot import plot_progression, plot_cluster_grid, plot_final_clusters, plot_kmeans_comparison

# ── Configurações ──────────────────────────────────────────────────
NUM_EPOCHS   = 10          # épocas de treinamento
GRID_M       = 5            # linhas do grid SOM
GRID_N       = 5            # colunas do grid SOM
LR           = 0.3          # taxa de aprendizado inicial
SIGMA        = 4            # raio de vizinhança inicial
N_CLUSTERS   = 5            # número de clusters K-Means (3 para Iris)
SAVE_DIR     = "src/acoPictures"   # diretório para salvar gráficos

def centroid_to_neuron_distance(centroids, som_weights):
    total = 0.0

    for c in centroids:
        best = float("inf")

        for w in som_weights:
            dist = np.linalg.norm(c - w)
            if dist < best:
                best = dist

        total += best

    return total / len(centroids)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("[1/7] Carregando dados (sem normalização)...")
    # carregar corretamente
    df = pd.read_csv("data/Dados.csv", header=None)

    # separar labels
    y_raw = df.iloc[:, -1]

    # mapear para inteiro
    species_map = {v: i for i, v in enumerate(sorted(y_raw.unique()))}
    species_names = [k for k, _ in sorted(species_map.items(), key=lambda x: x[1])]
    print(f"      Classes: {species_names}")
    species_int = y_raw.map(species_map).values
    print(species_int)
    
    print("[2/7] aplicando normalização...")
    # remover colunas indesejadas + label
    cols_to_drop = [5, 12, 13]
    X = df.drop(df.columns[cols_to_drop + [-1]], axis=1).values.astype(float)
    print(f"      {X.shape[0]} amostras | dim={X.shape[1]}")

    # labeled_df_class = df.drop(df.columns[cols_to_drop], axis=1)
    # print(labeled_df_class)

    # # FALTA NORMALIZAR OS DADOS
    # ###########

    # K-Means nos dados (BASELINE)
    print("[3/7] Rodando K-Means nos dados...")
    kmeans_data = SimpleKMeans(k=N_CLUSTERS, random_state=42)
    centroids, kmeans_labels = kmeans_data.fit(X)
    print(f"      Inertia (K-Means): {kmeans_data.inertia_:.4f}")

    # Treinar SOM 
    print(f"[4/7] Treinando SOM ({GRID_M}x{GRID_N}, {NUM_EPOCHS} épocas)...")
    dim = X.shape[1]
    origin = np.zeros(dim)

    som = SOM(
        line=GRID_M,
        column=GRID_N,
        learning_rate=LR,
        dimension=dim,
        initial_weights=origin,
        sigma=SIGMA
    )

    som.train(data=X, num_epoch=NUM_EPOCHS, labeled_data=(X, species_int))
    print("      Treinamento concluído!")

    print("[5/7] Plotando evolução dos neurônios por épocas...")
    total = len(som.history)
    indices_to_plot = sorted(set(
        [0] + list(range(9, total, 10)) + [total - 1]
    ))

    dim0, dim1 = 0, 1
    xlabel = "AT. região - Comprimento"
    ylabel = "AT. região - Largura"

    plot_progression(
        som.history, X, som, indices_to_plot, 
        dim0=dim0, dim1=dim1, xlabel=xlabel, ylabel=ylabel,
        save_dir=SAVE_DIR
    )

if __name__ == "__main__":
    main()