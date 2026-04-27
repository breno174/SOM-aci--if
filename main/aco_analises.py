import sys
import os
from datetime import datetime

# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from implement.som import SOM
from implement.kmenasSimple import KMeans as SimpleKMeans
from src.helper.plot import (
    plot_progression,
    plot_cluster_grid,
    plot_final_clusters,
    plot_kmeans_comparison,
)

# ── Configurações ──────────────────────────────────────────────────
NUM_EPOCHS = 100  # épocas de treinamento
GRID_M = 7  # linhas do grid SOM
GRID_N = 7  # colunas do grid SOM
LR = 0.4  # taxa de aprendizado inicial
SIGMA = 3  # raio de vizinhança inicial
N_CLUSTERS = 6  # número de clusters K-Means (3 para Iris)
SAVE_DIR = "src/acoPictures"  # diretório para salvar gráficos


def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)

    n_samples = len(X)
    indices = np.arange(n_samples)

    # embaralha índices (mantém X e y alinhados)
    np.random.shuffle(indices)

    split = int(n_samples * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


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


def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)

    n_samples = len(X)
    indices = np.arange(n_samples)

    # embaralha índices (mantém X e y alinhados)
    np.random.shuffle(indices)

    split = int(n_samples * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


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

    print("[2/7] aplicando normalização...")
    # remover colunas indesejadas + label
    cols_to_drop = [5, 12, 13]
    X = df.drop(df.columns[cols_to_drop + [-1]], axis=1).values.astype(float)
    print(f"      {X.shape[0]} amostras | dim={X.shape[1]}")

    # labeled_df_class = df.drop(df.columns[cols_to_drop], axis=1)
    # print(labeled_df_class)

    # Normalizando os dados (Z-score: diminui a média e divide pelo desvio padrão)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

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
        sigma=SIGMA,
    )
    # 1. separar
    X_train, X_test, y_train, y_test = train_test_split_manual(X, species_int)
    # 2. treinar SOM
    som.train(
        X_train, num_epoch=NUM_EPOCHS, X_test=X_test, y_train=y_train, y_test=y_test
    )
    print("      Treinamento concluído!")

    # ── [TESTE] Avaliação no conjunto de teste ────────────────────────
    print("\n[TESTE] Avaliando no conjunto de teste...")

    # 3a. Acurácia final
    acc = som.compute_accuracy(X_train, y_train, X_test, y_test)
    print(f"      Acurácia (teste): {acc:.2%}  ({int(acc * len(y_test))}/{len(y_test)} corretos)")

    # 3b. Matriz de confusão
    num_classes = len(species_names)
    conf_matrix = som.compute_confusion_matrix(X_test, y_test, num_classes)
    print("      Matriz de confusão (linhas=real, colunas=predito):")
    print(conf_matrix)

    # Salvar heatmap da matriz de confusão
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    im = ax_cm.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(species_names, rotation=45, ha="right", fontsize=9)
    ax_cm.set_yticklabels(species_names, fontsize=9)
    ax_cm.set_xlabel("Predito", fontsize=11)
    ax_cm.set_ylabel("Real", fontsize=11)
    ax_cm.set_title(f"Matriz de Confusão — Teste  (Acc={acc:.2%})", fontsize=13, fontweight="bold")
    # anotar cada célula com o valor
    for i in range(num_classes):
        for j in range(num_classes):
            val = conf_matrix[i, j]
            color = "white" if val > conf_matrix.max() * 0.6 else "black"
            ax_cm.text(j, i, str(val), ha="center", va="center", fontsize=9, color=color)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = f"{SAVE_DIR}/mental_confusion_matrix_{timestamp}.png"
    fig_cm.savefig(cm_path, dpi=120, bbox_inches="tight")
    print(f"      Matriz de confusão salva em: {cm_path}")
    plt.close(fig_cm)

    # 3c. Grid de neurônios rotulados pelo label dominante (usando dados de treino)
    label_grid, _ = som.label_nodes_by_data(X_train, y_train)
    fig_grid, ax_grid = plt.subplots(figsize=(6, 6))
    im2 = ax_grid.imshow(label_grid, cmap="tab10", vmin=0, vmax=num_classes - 1)
    cbar = fig_grid.colorbar(im2, ax=ax_grid, ticks=range(num_classes))
    cbar.ax.set_yticklabels(species_names, fontsize=8)
    # anotar cada neurônio
    for i in range(som.line):
        for j in range(som.column):
            lbl = label_grid[i, j]
            text = species_names[lbl] if lbl >= 0 else "?"
            ax_grid.text(j, i, text, ha="center", va="center", fontsize=7, color="white",
                         fontweight="bold")
    ax_grid.set_title("Grid SOM — Classe Dominante por Neurônio (treino)", fontsize=12, fontweight="bold")
    ax_grid.set_xlabel("Coluna")
    ax_grid.set_ylabel("Linha")
    plt.tight_layout()
    grid_path = f"{SAVE_DIR}/mental_label_grid_{timestamp}.png"
    fig_grid.savefig(grid_path, dpi=120, bbox_inches="tight")
    print(f"      Grid de neurônios salvo em: {grid_path}")
    plt.close(fig_grid)

    max_iterations = NUM_EPOCHS * len(X_train)

    # Juntar o estado inicial com os estados salvos a cada época
    initial_weights_snapshot = som.weights.copy()
    history_per_epoch = [initial_weights_snapshot.copy()] + som.history
    # 4. Salvar topologia do experimento
    experiment_path = f"src/experiments/aco_mental_{timestamp}.json"
    SOM.save_experiment(
        path=experiment_path,
        seed="42",
        initial_weights=initial_weights_snapshot,
        sample_order=[],  # Não mantemos a ordem de amostras de todas as épocas para simplificar
        config={
            "grid": [GRID_M, GRID_N],
            "dim": dim,
            "learning_rate": LR,
            "sigma": SIGMA,
            "num_epochs": NUM_EPOCHS,
            "num_samples": len(X_train),
        },
    )

    print("[5/7] Plotando evolução dos neurônios por épocas...")
    total = len(som.history)
    indices_to_plot = sorted(set([0] + list(range(9, total, 10)) + [total - 1]))

    # dim0, dim1 = 0, 1
    # xlabel = "AT. região - Comprimento"
    # ylabel = "AT. região - Largura"

    # plot_progression(
    #     som.history,
    #     X,
    #     som,
    #     indices_to_plot,
    #     dim0=dim0,
    #     dim1=dim1,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     save_dir=SAVE_DIR,
    # )

    som.plot_accuracy_history(save_path=f"{SAVE_DIR}/mental_accuracy_{timestamp}.png")
    som.plot_mse_history(save_path=f"{SAVE_DIR}/mental_eqm_{timestamp}.png")
    som.plot_neighborhood_decay(
        max_iterations=max_iterations,
        save_path=f"{SAVE_DIR}/mental_neighborhood_decay_{timestamp}.png",
    )


if __name__ == "__main__":
    main()
