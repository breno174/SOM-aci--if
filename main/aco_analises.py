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
NUM_EPOCHS = 150  # épocas de treinamento
GRID_M = 7  # linhas do grid SOM
GRID_N = 7  # colunas do grid SOM
LR = 0.3  # taxa de aprendizado inicial
SIGMA = 7  # raio de vizinhança inicial
N_CLUSTERS = 3  # número de clusters K-Means (3 macro-classes: Ótimo, Normal, Ruim)
SAVE_DIR = "src/single/acoPictures"  # diretório para salvar gráficos


def train_test_split_hierarchical(X, y_macro, y_raw, test_size=0.2, seed=42):
    np.random.seed(seed)

    train_idx = []
    test_idx = []

    subclasses = np.unique(y_raw)

    for c in subclasses:
        idx = np.where(y_raw == c)[0]
        np.random.shuffle(idx)

        n = len(idx)

        # split proporcional
        n_test = int(np.round(n * test_size))

        # garante presença mínima em ambos os conjuntos (se n >= 2)
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0

        n_train = n - n_test

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    y_raw_arr = np.array(y_raw)
    return (
        X[train_idx],
        X[test_idx],
        y_macro[train_idx],
        y_macro[test_idx],
        y_raw_arr[train_idx],
        y_raw_arr[test_idx],
    )


def train_test_split_fixed(X, y_macro, y_raw, seed=42):
    """
    Divisão solicitada para treino (total 175 amostras):
    - Ótimo (59 amostras): c1_p1 (59)
    - Normal (58 amostras): c2_p1 (58)
    - Ruim (58 amostras estratificadas pelas subclasses):
        c4_p1 (3), c4_p4 (3)
        c3_p1 (10), c3_p2 (10), c3_p3 (10), c3_p4 (10)
        c4_p2 (6), c4_p3 (6)
    O resto vai para o conjunto de teste.
    """
    np.random.seed(seed)

    train_idx = []
    test_idx = []

    # Dicionário com a quantidade exata de treino por sub-classe
    train_counts = {
        "c1_p1": 59,
        "c2_p1": 58,
        "c3_p1": 10,
        "c3_p2": 10,
        "c3_p3": 10,
        "c3_p4": 10,
        "c4_p2": 6,
        "c4_p3": 6,
        "c4_p1": 3,
        "c4_p4": 3
    }

    subclasses = np.unique(y_raw)

    for c in subclasses:
        idx = np.where(y_raw == c)[0]
        np.random.shuffle(idx)

        # Quantidade definida no dicionário (se não tiver, pega 0)
        n_train = train_counts.get(c, 0)
        
        # Garante que não vamos tentar pegar mais do que existe
        n_train = min(n_train, len(idx))

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Embaralhar os conjuntos finais
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    y_raw_arr = np.array(y_raw)
    return (
        X[train_idx],
        X[test_idx],
        y_macro[train_idx],
        y_macro[test_idx],
        y_raw_arr[train_idx],
        y_raw_arr[test_idx],
    )


def convert_raw_to_macro(y_raw_list):
    """Converte predições y_raw de volta para y_macro (0, 1, 2)"""
    macro_map = {
        "c1_p1": 0,
        "c2_p1": 1,
    }
    y_macro_pred = []
    for y in y_raw_list:
        if y in macro_map:
            y_macro_pred.append(macro_map[y])
        else:
            y_macro_pred.append(2)  # Tudo que não for c1_p1 ou c2_p1 é "Ruim" (2)
    return np.array(y_macro_pred)


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
    # coluna de dados classificados e proporção dos dados
    count_by_class = df.iloc[:, -1].value_counts()
    print(count_by_class)
    # separar labels
    y_raw = df.iloc[:, -1]

    # mapear para inteiro (classes originais)
    species_map_orig = {v: i for i, v in enumerate(sorted(y_raw.unique()))}
    species_names_orig = [
        k for k, _ in sorted(species_map_orig.items(), key=lambda x: x[1])
    ]
    print(f"      Classes originais: {species_names_orig}")

    # ── Agrupar em 3 macro-classes ──────────────────────────────────
    #   0 = Ótimo  (c1_p1)
    #   1 = Normal (c2_p1)
    #   2 = Ruim   (c3_* e c4_*)
    def _map_to_group(label: str) -> int:
        if label.startswith("c1"):
            return 0  # Ótimo
        elif label.startswith("c2"):
            return 1  # Normal
        else:
            return 2  # Ruim  (c3_*, c4_*, …)

    species_names = ["Ótimo", "Normal", "Ruim"]
    species_int = y_raw.map(_map_to_group).values
    print(f"      Macro-classes: {species_names}")
    print(
        f"      Distribuição: {dict(zip(*np.unique(species_int, return_counts=True)))}"
    )

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
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = (
        train_test_split_hierarchical(X, species_int, y_raw.values, test_size=0.2)
    )
    # 2. treinar SOM
    som.train(
        X_train, 
        num_epoch=NUM_EPOCHS, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test,
        early_stop=True,
        stop_window=10,
        stop_threshold=0.01
    )
    print("      Treinamento concluído!")

    # Aplicar rotulação baseada em y_raw
    som.fit_predict_labels(X_train, y_train_raw)

    # ── [TESTE] Avaliação no conjunto de teste ────────────────────────
    print("\n[TESTE] Avaliando no conjunto de teste...")

    # 3a. Acurácia final com k-BMUs
    y_pred_raw = som.predict_k_bmu(X_test, k=3)
    y_pred_macro = convert_raw_to_macro(y_pred_raw)

    correct = np.sum(y_pred_macro == y_test)
    acc = correct / len(y_test)
    print(
        f"      Acurácia (teste, 3-BMU): {acc:.2%}  ({correct}/{len(y_test)} corretos)"
    )

    # 3b. Matriz de confusão
    num_classes = len(species_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t_label, p_label in zip(y_test, y_pred_macro):
        conf_matrix[t_label][p_label] += 1

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
    ax_cm.set_title(
        f"Matriz de Confusão — Teste  (Acc={acc:.2%})", fontsize=13, fontweight="bold"
    )
    # anotar cada célula com o valor
    for i in range(num_classes):
        for j in range(num_classes):
            val = conf_matrix[i, j]
            color = "white" if val > conf_matrix.max() * 0.6 else "black"
            ax_cm.text(
                j, i, str(val), ha="center", va="center", fontsize=9, color=color
            )
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
            ax_grid.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
    ax_grid.set_title(
        "Grid SOM — Classe Dominante por Neurônio (treino)",
        fontsize=12,
        fontweight="bold",
    )
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
    experiment_path = f"src/experiments/single/aco_mental_{timestamp}.json"
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

    som.plot_accuracy_history(save_path=f"{SAVE_DIR}/mental_accuracy_{timestamp}.png")
    som.plot_mse_history(save_path=f"{SAVE_DIR}/mental_eqm_{timestamp}.png")
    som.plot_neighborhood_decay(
        max_iterations=max_iterations,
        save_path=f"{SAVE_DIR}/mental_neighborhood_decay_{timestamp}.png",
    )


if __name__ == "__main__":
    main()
