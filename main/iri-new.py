import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from implement.som import SOM
# from src.helper.kmeans import SimpleKMeans
from implement.kmenasSimple import KMeans as SimpleKMeans
from src.helper.plot import plot_progression, plot_cluster_grid, plot_final_clusters, plot_kmeans_comparison

# ── Configurações ──────────────────────────────────────────────────
NUM_EPOCHS   = 100          # épocas de treinamento
GRID_M       = 4            # linhas do grid SOM
GRID_N       = 4            # colunas do grid SOM
LR           = 0.3          # taxa de aprendizado inicial
SIGMA        = 2            # raio de vizinhança inicial
N_CLUSTERS   = 3            # número de clusters K-Means (3 para Iris)
SAVE_DIR     = "src/pictures"   # diretório para salvar gráficos

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

    # ── 1. Carregar dados (Sem Normalização) ────────────────────────
    print("[1/7] Carregando dados (sem normalização)...")
    df_full = pd.read_csv("data/Iris.csv")
    
    # Extrair features numéricas e converter para float
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    data = df_full[features].values.astype(float)
    
    print(f"      {data.shape[0]} amostras | dim={data.shape[1]}")

    print("[X] Rodando K-Means nos dados (baseline)...")

    # ── 2. Carregar labels de espécie ───────────────────────────────
    species_map = {v: i for i, v in enumerate(sorted(df_full['Species'].unique()))}
    species_int = df_full['Species'].map(species_map).values
    species_names = [k for k, _ in sorted(species_map.items(), key=lambda x: x[1])]
    print(f"      Classes: {species_names}")

    # ── 3. K-Means nos dados (BASELINE) ──────────────────
    print("[3/7] Rodando K-Means nos dados...")
    kmeans_data = SimpleKMeans(k=N_CLUSTERS, random_state=42)
    centroids, kmeans_labels = kmeans_data.fit(data)

    print(f"      Inertia (K-Means): {kmeans_data.inertia_:.4f}")

    # ── 4. Treinar SOM ───────────────────────────────────
    print(f"[3/8] Treinando SOM ({GRID_M}x{GRID_N}, {NUM_EPOCHS} épocas)...")

    dim = data.shape[1]
    origin = np.zeros(dim)

    som = SOM(
        line=GRID_M,
        column=GRID_N,
        learning_rate=LR,
        dimension=dim,
        initial_weights=origin,
        sigma=SIGMA
    )

    som.train(data=data, num_epoch=NUM_EPOCHS, labeled_data=(data, species_int))

    print("      Treinamento concluído!")

    # ── 4. Progressão dos neurônios por épocas ──────────────────────
    print("[4/8] Plotando evolução dos neurônios por épocas...")
    total = len(som.history)
    indices_to_plot = sorted(set(
        [0] + list(range(9, total, 10)) + [total - 1]
    ))
    
    # Vamos usar as features PetalLengthCm (2) e PetalWidthCm (3) para os gráficos de projeção
    dim0, dim1 = 2, 3
    xlabel = "PetalLengthCm"
    ylabel = "PetalWidthCm"
    
    plot_progression(
        som.history, data, som, indices_to_plot, 
        dim0=dim0, dim1=dim1, xlabel=xlabel, ylabel=ylabel
    )

    # ── 5. Validação SOM vs K-Means ──────────────────────
    print("[5/8] Validando SOM...")

    qe = som.quantization_error(data)
    print(f"      Quantization Error (SOM): {qe:.4f}")

    dist = centroid_to_neuron_distance(centroids, som.weights)
    print(f"      Distancia centroide -> neuronio: {dist:.4f}")

    epocas = list(range(1, len(som.history_mse) + 1))

    # ── 6. Decaimento do LR e Sigma ─────────────────────────────────
    print("[5/7] Plotando decaimento dos parâmetros...")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Decaimento dos Parâmetros da SOM ao Longo das Épocas", fontsize=13, fontweight='bold')

    ax[0].plot(epocas, som.history_sigma, color='darkorange', linewidth=2, marker='o', markersize=2)
    # ax[0].fill_between(epocas, som.history_sigma, alpha=0.15, color='darkorange')
    ax[0].set_title("Raio de Vizinhança (σ)", fontweight='bold')
    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("σ")
    ax[0].grid(True, linestyle='--', alpha=0.4)

    ax[1].plot(epocas, som.history_lr, color='purple', linewidth=2, marker='o', markersize=2)
    # ax[1].fill_between(epocas, som.history_lr, alpha=0.15, color='purple')
    ax[1].set_title("Taxa de Aprendizado (α)", fontweight='bold')
    ax[1].set_xlabel("Época")
    ax[1].set_ylabel("α")
    ax[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/iris_neighborhood_decay.png", dpi=120, bbox_inches='tight')
    plt.show()

    # ── 7. Métricas (EQM / Acurácia) ─────────────────────
    print("[6/8] Plotando métricas...")

    epocas = list(range(1, len(som.history_mse) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epocas, som.history_mse, color='crimson', marker='o', markersize=3, linewidth=2, label="EQM médio")
    ax.set_title("EQM por Época")
    ax.set_xlabel("Época")
    ax.set_ylabel("EQM médio")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/iris_eqm.png", dpi=120, bbox_inches='tight')
    plt.show()

    if som.history_accuracy:
        fig, ax = plt.subplots(figsize=(10, 4))
        acc_pct = [a * 100 for a in som.history_accuracy]
        ax.plot(epocas, acc_pct, color='steelblue', marker='s', markersize=3, linewidth=2, label="Acurácia (%)")
        ax.set_xlabel("Época")
        ax.set_ylabel("Acurácia (%)")
        ax.set_title("Acurácia (%)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/iris_accuracy.png", dpi=120, bbox_inches='tight')
        plt.show()

    # ── 7. K-Means nos neurônios ──────────────────────────
    print("[7/8] Clustering dos neurônios (visualização)...")

    kmeans_neurons = SimpleKMeans(k=N_CLUSTERS, random_state=42)
    neuron_centroids, neuron_labels = kmeans_neurons.fit(som.weights)

    # ── 7b. Comparacao K-Means: dados brutos vs neuronios ──────────
    print("[7b/8] Plotando comparacao K-Means (dados vs neuronios)...")
    plot_kmeans_comparison(
        som=som,
        data=data,
        kmeans_data=kmeans_data,
        kmeans_data_labels=kmeans_labels,
        kmeans_neurons=kmeans_neurons,
        neuron_labels=neuron_labels,
        true_labels=species_int,
        class_names=species_names,
        dim0=dim0,
        dim1=dim1,
        xlabel=xlabel,
        ylabel=ylabel
    )

    # ── 8. Grade de clusters e labels ───────────────────────────────
    print("[8/8] Plotando grade de clusters e labels...")
    plot_cluster_grid(som, neuron_labels, species_int, data, species_names)

    print("\n[OK] Todos os graficos foram gerados com sucesso para Iris!")

if __name__ == "__main__":
    main()