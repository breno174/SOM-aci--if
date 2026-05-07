import sys
import os
import json
import random
from datetime import datetime
from itertools import product

# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from implement.som import SOM
from implement.kmenasSimple import KMeans as SimpleKMeans
from main.aco_analises import (
    train_test_split_hierarchical,
    train_test_split_fixed,
    convert_raw_to_macro,
)

# ── Configurações fixas ────────────────────────────────────────────
NUM_EPOCHS = 100  # épocas de treinamento
N_CLUSTERS = 5  # número de clusters K-Means (3 macro-classes: Ótimo, Normal, Ruim)
K_BMU = 3  # número de BMUs para predição
SAVE_DIR = "src/acoPictures/multi"  # diretório para salvar gráficos
EXPERIMENT_DIR = "src/experiments/multi"  # diretório para salvar experimentos

# ── Ranges para busca de topologia ─────────────────────────────────
GRID_M_VALUES = [5, 7, 9]  # linhas do grid SOM
GRID_N_VALUES = [5, 7, 9]  # colunas do grid SOM
LR_VALUES = [0.2, 0.3, 0.4, 0.5]  # taxa de aprendizado inicial
SIGMA_VALUES = [3, 5, 7, 9]  # raio de vizinhança inicial
NUM_RUNS_PER_TOPO = 5  # treinos independentes por topologia
MAX_TOTAL_RUNS = 100  # limite máximo de treinos no total


def generate_topologies():
    """
    Gera todas as combinações válidas de (GRID_M, GRID_N, LR, SIGMA).
    Regra: SIGMA não pode ser maior que GRID_M * GRID_N.
    """
    topologies = []
    for m, n, lr, sigma in product(
        GRID_M_VALUES, GRID_N_VALUES, LR_VALUES, SIGMA_VALUES
    ):
        if sigma <= m * n:
            topologies.append({"grid_m": m, "grid_n": n, "lr": lr, "sigma": sigma})
    return topologies


def load_and_prepare_data():
    """Carrega e pré-processa os dados uma única vez."""
    print("[1] Carregando dados...")
    df = pd.read_csv("data/Dados.csv", header=None)

    count_by_class = df.iloc[:, -1].value_counts()
    print(count_by_class)

    y_raw = df.iloc[:, -1]

    # ── Agrupar em 3 macro-classes ──────────────────────────────────
    def _map_to_group(label: str) -> int:
        if label.startswith("c1"):
            return 0  # Ótimo
        elif label.startswith("c2"):
            return 1  # Normal
        else:
            return 2  # Ruim

    species_names = ["Ótimo", "Normal", "Ruim"]
    species_int = y_raw.map(_map_to_group).values
    print(f"   Macro-classes: {species_names}")
    print(f"   Distribuição: {dict(zip(*np.unique(species_int, return_counts=True)))}")

    # remover colunas indesejadas + label
    cols_to_drop = [5, 12, 13, 0, 3, 6, 15, 16, 17, 18, 19, 22, 23]
    all_cols = list(range(df.shape[1] - 1))  # excluindo label
    valid_cols = [c for c in all_cols if c not in cols_to_drop]
    feature_names = [f"F_{c}" for c in valid_cols]

    X = df.drop(df.columns[cols_to_drop + [-1]], axis=1).values.astype(float)
    print(f"   {X.shape[0]} amostras | dim={X.shape[1]}")

    # Normalizando os dados (Z-score)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, species_int, y_raw.values, species_names, feature_names


def run_single_training(X, species_int, y_raw_values, seed, grid_m, grid_n, lr, sigma):
    """Executa um treino completo do SOM e retorna métricas."""
    dim = X.shape[1]
    origin = np.zeros(dim)

    som = SOM(
        line=grid_m,
        column=grid_n,
        learning_rate=lr,
        dimension=dim,
        initial_weights=origin,
        sigma=sigma,
    )
    initial_weights_snapshot = som.weights.copy()

    # 1. separar dados com seed diferente para cada rodada
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = (
        train_test_split_hierarchical(
            X, species_int, y_raw_values, test_size=0.3, seed=seed
        )
    )

    # X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split_fixed(
    #     X, species_int, y_raw_values, seed=seed
    # )

    # 2. treinar SOM com early stopping ativado
    som.train(
        X_train,
        num_epoch=NUM_EPOCHS,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        early_stop=True,
        stop_window=10,
        stop_threshold=0.01,
    )

    # 3. Rotulação baseada em y_raw
    som.fit_predict_labels(X_train, y_train_raw)

    # 4. Predição com k-BMUs no teste
    y_pred_raw_test = som.predict_k_bmu(X_test, k=K_BMU)
    y_pred_macro_test = convert_raw_to_macro(y_pred_raw_test)
    acc_test = np.sum(y_pred_macro_test == y_test) / len(y_test)

    # 5. Predição com k-BMUs no treino (para calcular acurácia de treino)
    y_pred_raw_train = som.predict_k_bmu(X_train, k=K_BMU)
    y_pred_macro_train = convert_raw_to_macro(y_pred_raw_train)
    acc_train = np.sum(y_pred_macro_train == y_train) / len(y_train)

    # 6. EQM de treino e teste
    mse_train_history = som.history_mse  # EQM por época (treino)

    # EQM por época no teste (calculado retroativamente com os snapshots de pesos)
    mse_test_history = []
    for epoch_weights in som.history:
        total_sq = 0.0
        for x in X_test:
            diffs = epoch_weights - x
            dists = np.sum(diffs**2, axis=1)
            total_sq += np.min(dists)
        mse_test_history.append(total_sq / len(X_test))

    # Acurácia por época (no teste, já calculada durante o treino)
    acc_history_test = som.history_accuracy

    # Matriz de confusão (teste)
    num_classes = 3
    conf_matrix_test = np.zeros((num_classes, num_classes), dtype=int)
    for t_label, p_label in zip(y_test, y_pred_macro_test):
        conf_matrix_test[t_label][p_label] += 1

    # Matriz de confusão (treino)
    conf_matrix_train = np.zeros((num_classes, num_classes), dtype=int)
    for t_label, p_label in zip(y_train, y_pred_macro_train):
        conf_matrix_train[t_label][p_label] += 1

    return {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "mse_train_history": mse_train_history,
        "mse_test_history": mse_test_history,
        "acc_history_test": acc_history_test,
        "conf_matrix_test": conf_matrix_test,
        "conf_matrix_train": conf_matrix_train,
        "history_sigma": som.history_sigma,
        "history_lr": som.history_lr,
        "som": som,
        "initial_weights": initial_weights_snapshot,
        "num_train_samples": len(X_train),
        "y_test_raw": y_test_raw,
        "y_pred_macro_test": y_pred_macro_test,
        "y_pred_raw_test": y_pred_raw_test,
        "y_test_macro": y_test,
        "X_test": X_test,
    }


def plot_accuracy_comparison(
    all_results, species_names, save_path, topology_text=None, x_labels=None
):
    """
    Gráfico 1: Acurácia média de treino vs teste por rodada,
    com linha de média geral e suavização por média móvel.
    """
    n_runs = len(all_results)
    acc_trains = [r["acc_train"] * 100 for r in all_results]
    acc_tests = [r["acc_test"] * 100 for r in all_results]

    x = np.arange(1, n_runs + 1)

    # Tamanho fixo adequado para slides (aprox 16:9), sem alargar ao infinito
    fig, ax = plt.subplots(figsize=(14, 6))

    # Linhas originais mais claras (reduz o ruído visual, mas mantém a informação)
    ax.plot(
        x,
        acc_trains,
        color="#2196F3",
        linewidth=1,
        alpha=0.3,
        linestyle="--",
    )
    ax.plot(
        x,
        acc_tests,
        color="#F44336",
        linewidth=1,
        alpha=0.3,
        linestyle="--",
    )

    # Média móvel (janela de 5)
    window_size = 5 if n_runs >= 5 else 1
    train_smooth = (
        pd.Series(acc_trains).rolling(window=window_size, min_periods=1).mean()
    )
    test_smooth = pd.Series(acc_tests).rolling(window=window_size, min_periods=1).mean()

    ax.plot(
        x,
        train_smooth,
        color="#2196F3",
        linewidth=2.5,
        marker="o" if n_runs <= 20 else None,
        markersize=6 if n_runs <= 20 else 0,
        label=f"Treino (Média Móvel {window_size})" if window_size > 1 else "Treino",
    )
    ax.plot(
        x,
        test_smooth,
        color="#F44336",
        linewidth=2.5,
        marker="s" if n_runs <= 20 else None,
        markersize=6 if n_runs <= 20 else 0,
        label=f"Teste (Média Móvel {window_size})" if window_size > 1 else "Teste",
    )

    mean_train = np.mean(acc_trains)
    mean_test = np.mean(acc_tests)

    ax.axhline(
        y=mean_train,
        color="#2196F3",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Média Geral Treino ({mean_train:.2f}%)",
    )
    ax.axhline(
        y=mean_test,
        color="#F44336",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Média Geral Teste ({mean_test:.2f}%)",
    )

    ax.set_title(
        f"Evolução da Acurácia por Rodada — {n_runs} Treinos ({K_BMU}-BMU)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rodada", fontsize=12)
    ax.set_ylabel("Acurácia (%)", fontsize=12)

    if x_labels is not None:
        if n_runs > 30:
            # Mostrar menos rótulos para caber no slide sem poluir
            step = max(1, n_runs // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(
                [x_labels[i] for i in range(0, n_runs, step)],
                rotation=45,
                ha="right",
                fontsize=9,
            )
        else:
            ax.set_xticks(x)
            rotation_angle = 90 if n_runs > 15 else 45
            ha_align = "center" if n_runs > 15 else "right"
            ax.set_xticklabels(
                x_labels, rotation=rotation_angle, ha=ha_align, fontsize=9
            )
    else:
        ax.set_xticks(x)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    if topology_text:
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        ax.text(
            1.01,
            1.0,
            topology_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )
        plt.subplots_adjust(right=0.85)
    else:
        plt.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de acurácia salvo em: {save_path}")
    plt.close(fig)


def plot_sorted_accuracies(all_results, save_path, x_labels=None, custom_title=None):
    """
    Gráfico Extra: Gráfico de barras horizontais ordenado pela Acurácia de Teste.
    Ideal para visualizar muitos treinos de uma vez.
    """
    n_runs = len(all_results)

    # Agrupar dados
    data = []
    for i, r in enumerate(all_results):
        label = x_labels[i] if x_labels else f"Rodada {i+1}"
        data.append((r["acc_test"], r["acc_train"], label))

    # Ordenar por acurácia de teste
    data.sort(key=lambda x: x[0])

    acc_tests = [d[0] * 100 for d in data]
    acc_trains = [d[1] * 100 for d in data]
    labels = [d[2] for d in data]

    # Altura dinâmica para acomodar muitos resultados
    fig_height = max(6, n_runs * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y = np.arange(n_runs)
    height = 0.4

    ax.barh(y - height / 2, acc_trains, height, color="#2196F3", label="Treino")
    ax.barh(y + height / 2, acc_tests, height, color="#F44336", label="Teste")

    title_str = (
        custom_title if custom_title else f"Ranking de Acurácia — {n_runs} Treinos"
    )
    ax.set_title(title_str, fontsize=13, fontweight="bold")
    ax.set_xlabel("Acurácia (%)", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Limitar x até um pouco mais que o máximo para caber o texto
    max_acc = max(max(acc_tests), max(acc_trains))
    ax.set_xlim(0, min(100 + 10, max_acc + 10))

    # Adicionar textos nas barras de teste e treino
    for i in range(n_runs):
        val_test = acc_tests[i]
        val_train = acc_trains[i]

        # Texto para Teste
        ax.text(
            val_test + 0.5,
            y[i] + height / 2,
            f"{val_test:.1f}%",
            va="center",
            fontsize=7,
            color="#F44336",
            fontweight="bold",
        )

        # Texto para Treino
        ax.text(
            val_train + 0.5,
            y[i] - height / 2,
            f"{val_train:.1f}%",
            va="center",
            fontsize=7,
            color="#2196F3",
            fontweight="bold",
        )

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de ranking salvo em: {save_path}")
    plt.close(fig)


def plot_mse_comparison(all_results, save_path):
    """
    Gráfico 2: Dois subplots lado a lado.
    - Esquerda: EQM médio por época de TREINO (média de todas as rodadas).
    - Direita: EQM médio por época de TESTE (média de todas as rodadas).
    """
    n_runs = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_epochs = max(len(r["mse_train_history"]) for r in all_results)
    epochs = list(range(1, max_epochs + 1))

    # ── Esquerda: EQM médio de TREINO ──
    raw_mse_train = [r["mse_train_history"] for r in all_results]
    all_mse_train = np.array(
        [np.pad(arr, (0, max_epochs - len(arr)), mode="edge") for arr in raw_mse_train]
    )
    mean_train = np.mean(all_mse_train, axis=0)
    std_train = np.std(all_mse_train, axis=0)

    axes[0].plot(
        epochs,
        mean_train,
        color="purple",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"EQM Treino (média de {n_runs} rodadas)",
    )
    axes[0].fill_between(
        epochs,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2,
        color="purple",
        label="± 1 Desvio Padrão",
    )
    axes[0].set_title(
        f"Evolução do EQM de Treino (média de {n_runs} rodadas)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("EQM", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    # ── Direita: EQM médio de TESTE ──
    raw_mse_test = [r["mse_test_history"] for r in all_results]
    all_mse_test = np.array(
        [np.pad(arr, (0, max_epochs - len(arr)), mode="edge") for arr in raw_mse_test]
    )
    mean_test = np.mean(all_mse_test, axis=0)
    std_test = np.std(all_mse_test, axis=0)

    axes[1].plot(
        epochs,
        mean_test,
        color="#F44336",
        linewidth=2,
        marker="s",
        markersize=3,
        label=f"EQM Teste (média de {n_runs} rodadas)",
    )
    axes[1].fill_between(
        epochs,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2,
        color="#F44336",
        label="± 1 Desvio Padrão",
    )
    axes[1].set_title(
        f"Evolução do EQM de Teste (média de {n_runs} rodadas)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("EQM", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de EQM salvo em: {save_path}")
    plt.close(fig)


def plot_confusion_matrix_mean(
    all_results,
    species_names,
    save_path,
    custom_title_prefix="Matriz de Confusão Acumulada",
    subtitle_suffix=None,
):
    """
    Gráfico 3: Duas matrizes de confusão.
    Esquerda = Treino, Direita = Teste.
    """
    num_classes = len(species_names)
    n_runs = len(all_results)

    total_train = np.zeros((num_classes, num_classes), dtype=int)
    total_test = np.zeros((num_classes, num_classes), dtype=int)

    for r in all_results:
        total_train += r["conf_matrix_train"]
        total_test += r["conf_matrix_test"]

    acc_train = (
        np.trace(total_train) / np.sum(total_train) if np.sum(total_train) > 0 else 0
    )
    acc_test = (
        np.trace(total_test) / np.sum(total_test) if np.sum(total_test) > 0 else 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, matrix, title, acc in [
        (axes[0], total_train, f"{custom_title_prefix} — TREINO", acc_train),
        (axes[1], total_test, f"{custom_title_prefix} — TESTE", acc_test),
    ]:
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(species_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(species_names, fontsize=10)
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        subtitle = subtitle_suffix if subtitle_suffix else f"{n_runs} rodadas"
        ax.set_title(
            f"{title}\n{subtitle} (Acc={acc:.2%})",
            fontsize=12,
            fontweight="bold",
        )

        for i in range(num_classes):
            for j in range(num_classes):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                    fontweight="bold",
                )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Matriz de confusão salva em: {save_path}")
    plt.close(fig)


def plot_decay_comparison(all_results, save_path):
    """
    Gráfico 4: Decaimento médio do raio de vizinhança (σ) e da taxa de
    aprendizado (α) ao longo das épocas, considerando todas as rodadas.
    """
    n_runs = len(all_results)

    raw_sigma = [r["history_sigma"] for r in all_results]
    max_len_sigma = max(len(arr) for arr in raw_sigma)
    all_sigma = np.array(
        [np.pad(arr, (0, max_len_sigma - len(arr)), mode="edge") for arr in raw_sigma]
    )

    raw_lr = [r["history_lr"] for r in all_results]
    max_len_lr = max(len(arr) for arr in raw_lr)
    all_lr = np.array(
        [np.pad(arr, (0, max_len_lr - len(arr)), mode="edge") for arr in raw_lr]
    )

    mean_sigma = np.mean(all_sigma, axis=0)
    std_sigma = np.std(all_sigma, axis=0)
    mean_lr = np.mean(all_lr, axis=0)
    std_lr = np.std(all_lr, axis=0)

    epochs = list(range(1, len(mean_sigma) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Decaimento dos Parâmetros da SOM (média de {n_runs} rodadas)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Esquerda: Sigma ──
    axes[0].plot(
        epochs,
        mean_sigma,
        color="darkorange",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"σ médio ({n_runs} rodadas)",
    )
    axes[0].fill_between(
        epochs,
        mean_sigma - std_sigma,
        mean_sigma + std_sigma,
        alpha=0.2,
        color="darkorange",
        label="± 1 Desvio Padrão",
    )
    axes[0].set_title("Decaimento do Raio de Vizinhança (σ)", fontweight="bold")
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("σ (sigma)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    # ── Direita: Learning Rate ──
    axes[1].plot(
        epochs,
        mean_lr,
        color="purple",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"α médio ({n_runs} rodadas)",
    )
    axes[1].fill_between(
        epochs,
        mean_lr - std_lr,
        mean_lr + std_lr,
        alpha=0.2,
        color="purple",
        label="± 1 Desvio Padrão",
    )
    axes[1].set_title("Decaimento da Taxa de Aprendizado (α)", fontweight="bold")
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("α (learning rate)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de decaimento salvo em: {save_path}")
    plt.close(fig)


def plot_synoptic_map(som, species_names, save_path, title_suffix=""):
    """
    Gráfico: Mapa Sinótico — mostra a classe dominante de cada neurônio
    no grid da SOM, colorido por macro-classe.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    grid_m = som.line
    grid_n = som.column
    num_classes = len(species_names)

    # Construir grid de macro-classes a partir dos rótulos raw do SOM
    label_grid = np.full((grid_m, grid_n), -1, dtype=int)
    for idx in range(som.num_neurons):
        raw_label = som.neuron_majority_label.get(idx)
        if raw_label is not None:
            macro = convert_raw_to_macro([raw_label])[0]
            row = idx // grid_n
            col = idx % grid_n
            label_grid[row, col] = macro

    # Cores distintas: Ótimo=verde, Normal=azul, Ruim=vermelho, vazio=cinza
    class_colors = ["#4CAF50", "#2196F3", "#F44336"]
    empty_color = "#E0E0E0"

    # Criar colormap com slot para -1 (vazio)
    all_colors = [empty_color] + class_colors
    cmap = ListedColormap(all_colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    cell_size = 0.9
    fig_w = max(5, grid_n * cell_size + 2)
    fig_h = max(4, grid_m * cell_size + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(label_grid, cmap=cmap, norm=norm, aspect="equal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(num_classes)), shrink=0.8, pad=0.02)
    cbar.ax.set_yticklabels(species_names, fontsize=9)

    # Anotar cada neurônio com o nome da classe
    for i in range(grid_m):
        for j in range(grid_n):
            lbl = label_grid[i, j]
            if lbl >= 0:
                text = species_names[lbl]
                # Texto branco sobre fundo colorido
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )
            else:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#999999",
                )

    # Linhas de grade
    ax.set_xticks(np.arange(-0.5, grid_n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_m, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks(range(grid_n))
    ax.set_yticks(range(grid_m))
    ax.set_xlabel("Coluna", fontsize=11)
    ax.set_ylabel("Linha", fontsize=11)

    title = f"Mapa Sinótico — Classe Dominante por Neurônio"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Mapa sinótico salvo em: {save_path}")
    plt.close(fig)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carregar dados uma vez
    X, species_int, y_raw_values, species_names, feature_names = load_and_prepare_data()

    # ── Gerar topologias ───────────────────────────────────────────
    topologies = generate_topologies()

    # Embaralhar para escolher topologias aleatoriamente
    random.shuffle(topologies)

    # Limitar o número de topologias para não ultrapassar MAX_TOTAL_RUNS
    max_topos = max(1, MAX_TOTAL_RUNS // NUM_RUNS_PER_TOPO)
    if len(topologies) > max_topos:
        topologies = topologies[:max_topos]

    total_topos = len(topologies)
    print(f"\n[INFO] {total_topos} topologias selecionadas aleatoriamente.")
    print(
        f"[INFO] {NUM_RUNS_PER_TOPO} treinos por topologia → {total_topos * NUM_RUNS_PER_TOPO} treinos totais."
    )

    # Armazena resumo de cada topologia
    topo_summaries = []  # lista de dicts com topologia + métricas médias
    all_topo_results = {}  # dict: topo_idx -> lista de results (para gráficos)
    global_run = 0  # contador global de rodadas

    best_topo_idx = -1
    best_mean_acc_test = -1.0

    for topo_idx, topo in enumerate(topologies):
        grid_m = topo["grid_m"]
        grid_n = topo["grid_n"]
        lr = topo["lr"]
        sigma = topo["sigma"]

        print(f"\n{'#'*60}")
        print(f"  TOPOLOGIA {topo_idx + 1}/{total_topos}")
        print(f"  GRID={grid_m}x{grid_n} | LR={lr} | SIGMA={sigma}")
        print(f"{'#'*60}")

        topo_results = []

        for run in range(NUM_RUNS_PER_TOPO):
            global_run += 1
            seed = 42 + run
            print(f"\n{'='*60}")
            print(
                f"  Topo {topo_idx+1} — Rodada {run+1}/{NUM_RUNS_PER_TOPO}  (seed={seed})"
            )
            print(f"{'='*60}")

            result = run_single_training(
                X,
                species_int,
                y_raw_values,
                seed=seed,
                grid_m=grid_m,
                grid_n=grid_n,
                lr=lr,
                sigma=sigma,
            )

            result["topo_idx"] = topo_idx + 1
            result["run_idx"] = run + 1
            result["grid_m"] = grid_m
            result["grid_n"] = grid_n
            result["lr"] = lr
            result["sigma"] = sigma

            print(f"   Acc Treino: {result['acc_train']:.2%}")
            print(f"   Acc Teste:  {result['acc_test']:.2%}")
            print(f"   EQM Final (treino): {result['mse_train_history'][-1]:.4f}")
            print(f"   EQM Final (teste):  {result['mse_test_history'][-1]:.4f}")

            # Salvar experimento da rodada
            experiment_path = (
                f"{EXPERIMENT_DIR}/aco_topo{topo_idx+1}_run{run+1}_{timestamp}.json"
            )
            SOM.save_experiment(
                path=experiment_path,
                seed=seed,
                initial_weights=result["initial_weights"],
                sample_order=[],
                config={
                    "grid": [grid_m, grid_n],
                    "dim": X.shape[1],
                    "learning_rate": lr,
                    "sigma": sigma,
                    "num_epochs": NUM_EPOCHS,
                    "num_samples": result["num_train_samples"],
                    "k_bmu": K_BMU,
                    "topology_index": topo_idx + 1,
                    "run": run + 1,
                    "acc_train": float(result["acc_train"]),
                    "acc_test": float(result["acc_test"]),
                },
            )

            topo_results.append(result)

        # ── Resumo desta topologia ─────────────────────────────────
        acc_trains = [r["acc_train"] for r in topo_results]
        acc_tests = [r["acc_test"] for r in topo_results]
        mean_acc_train = float(np.mean(acc_trains))
        mean_acc_test = float(np.mean(acc_tests))
        std_acc_train = float(np.std(acc_trains))
        std_acc_test = float(np.std(acc_tests))

        mean_final_lr = float(np.mean([r["history_lr"][-1] for r in topo_results]))
        mean_final_sigma = float(
            np.mean([r["history_sigma"][-1] for r in topo_results])
        )

        topo_summary = {
            "topology_index": topo_idx + 1,
            "grid_m": grid_m,
            "grid_n": grid_n,
            "lr": lr,
            "sigma": sigma,
            "final_lr": mean_final_lr,
            "final_sigma": mean_final_sigma,
            "num_runs": NUM_RUNS_PER_TOPO,
            "mean_acc_train": mean_acc_train,
            "std_acc_train": std_acc_train,
            "mean_acc_test": mean_acc_test,
            "std_acc_test": std_acc_test,
            "best_acc_test": float(np.max(acc_tests)),
            "worst_acc_test": float(np.min(acc_tests)),
        }
        topo_summaries.append(topo_summary)

        print(
            f"\n  → Média Treino: {mean_acc_train*100:.2f}% ± {std_acc_train*100:.2f}%"
        )
        print(f"  → Média Teste:  {mean_acc_test*100:.2f}% ± {std_acc_test*100:.2f}%")

        # Guardar resultados desta topologia para gráficos posteriores
        all_topo_results[topo_idx] = topo_results

        # Verificar se é a melhor topologia até agora
        if mean_acc_test > best_mean_acc_test:
            best_mean_acc_test = mean_acc_test
            best_topo_idx = topo_idx

    # ── Resumo geral ──────────────────────────────────────────────
    best = topo_summaries[best_topo_idx]

    print(f"\n{'='*60}")
    print(f"  RESUMO GERAL — {total_topos} TOPOLOGIAS × {NUM_RUNS_PER_TOPO} TREINOS")
    print(f"{'='*60}")
    print(f"\n  🏆 MELHOR TOPOLOGIA (por acurácia média de teste):")
    print(f"     Topologia #{best['topology_index']}")
    print(f"     GRID = {best['grid_m']}×{best['grid_n']}")
    print(f"     LR (Inicial -> Final) = {best['lr']} -> {best['final_lr']:.4f}")
    print(
        f"     SIGMA (Inicial -> Final) = {best['sigma']} -> {best['final_sigma']:.4f}"
    )
    print(
        f"     Acc Teste Média:  {best['mean_acc_test']*100:.2f}% ± {best['std_acc_test']*100:.2f}%"
    )
    print(
        f"     Acc Treino Média: {best['mean_acc_train']*100:.2f}% ± {best['std_acc_train']*100:.2f}%"
    )
    print(f"     Melhor Teste:     {best['best_acc_test']*100:.2f}%")
    print(f"     Pior Teste:       {best['worst_acc_test']*100:.2f}%")

    # ── Salvar JSON com todas as topologias e a melhor ─────────────
    summary_json = {
        "timestamp": timestamp,
        "num_epochs": NUM_EPOCHS,
        "k_bmu": K_BMU,
        "num_runs_per_topology": NUM_RUNS_PER_TOPO,
        "total_topologies": total_topos,
        "best_topology": best,
        "all_topologies": topo_summaries,
    }
    summary_path = f"{EXPERIMENT_DIR}/topology_search_{timestamp}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"\n   📄 Resumo salvo em: {summary_path}")

    # ── Gerar gráficos com TODOS os resultados ──────────────────────
    all_results = []
    topo_legends = []
    for topo_idx_key in sorted(all_topo_results.keys()):
        all_results.extend(all_topo_results[topo_idx_key])
        summary = next(
            t for t in topo_summaries if t["topology_index"] == topo_idx_key + 1
        )
        topo_legends.append(
            f"T{summary['topology_index']}: {summary['grid_m']}x{summary['grid_n']}, "
            f"LR={summary['lr']}, σ={summary['sigma']}"
        )

    print(
        f"\n[GRÁFICOS] Gerando visualizações agregadas ({len(all_results)} treinos no total)..."
    )

    x_labels_multi = [f"T{r['topo_idx']}-{r['run_idx']}" for r in all_results]
    topology_text_multi = "Legenda de Topologias:\n" + "\n".join(topo_legends)

    # 1. Acurácia treino vs teste por rodada
    plot_accuracy_comparison(
        all_results,
        species_names,
        save_path=f"{SAVE_DIR}/multi_accuracy_{timestamp}.png",
        topology_text=topology_text_multi,
        x_labels=x_labels_multi,
    )

    # 1.5 Gráfico de barras ordenado para visualizar ranking de todas as rodadas
    if len(all_results) > 1:
        plot_sorted_accuracies(
            all_results,
            save_path=f"{SAVE_DIR}/multi_accuracy_ranking_{timestamp}.png",
            x_labels=x_labels_multi,
        )

    # 1.6 Gráfico de barras ordenado (apenas Top 5 Geral)
    if len(all_results) >= 5:
        # Criar pares de (resultado, label_correspondente) e ordenar
        paired_results = list(zip(all_results, x_labels_multi))
        paired_results.sort(key=lambda item: item[0]["acc_test"], reverse=True)
        top5_pairs = paired_results[:5]

        top5_results = [p[0] for p in top5_pairs]
        top5_labels = [p[1] for p in top5_pairs]

        plot_sorted_accuracies(
            top5_results,
            save_path=f"{SAVE_DIR}/top5_accuracy_ranking_{timestamp}.png",
            x_labels=top5_labels,
            custom_title="Ranking de Acurácia — Top 5 Melhores Treinos",
        )

    # 2. EQM: média de treino e teste
    plot_mse_comparison(
        all_results,
        save_path=f"{SAVE_DIR}/multi_eqm_{timestamp}.png",
    )

    # 3. Matriz de confusão acumulada
    plot_confusion_matrix_mean(
        all_results,
        species_names,
        save_path=f"{SAVE_DIR}/multi_confusion_matrix_{timestamp}.png",
    )

    # 4. Decaimento médio de sigma e learning rate
    plot_decay_comparison(
        all_results,
        save_path=f"{SAVE_DIR}/multi_decay_{timestamp}.png",
    )

    # ── Gerar gráficos para a MELHOR topologia ──────────────────────
    best_results = all_topo_results[best_topo_idx]

    print(
        f"\n[GRÁFICOS] Gerando visualizações para a MELHOR topologia ({len(best_results)} treinos)..."
    )

    # 1. Acurácia treino vs teste por rodada
    best_topo_text = (
        f"Melhor Topologia (T{best['topology_index']}):\n"
        f"Grid: {best['grid_m']}x{best['grid_n']}\n"
        f"Learning Rate: {best['lr']}\n"
        f"Sigma: {best['sigma']}"
    )
    x_labels_best = [f"R{r['run_idx']}" for r in best_results]

    plot_accuracy_comparison(
        best_results,
        species_names,
        save_path=f"{SAVE_DIR}/best_topo_accuracy_{timestamp}.png",
        topology_text=best_topo_text,
        x_labels=x_labels_best,
    )

    if len(best_results) > 1:
        plot_sorted_accuracies(
            best_results,
            save_path=f"{SAVE_DIR}/best_topo_accuracy_ranking_{timestamp}.png",
            x_labels=x_labels_best,
        )

    # 2. EQM: média de treino e teste
    plot_mse_comparison(
        best_results,
        save_path=f"{SAVE_DIR}/best_topo_eqm_{timestamp}.png",
    )

    # 3. Matriz de confusão da melhor rodada (média da acc treino e teste)
    best_single_run = max(
        best_results, key=lambda r: (r["acc_train"] + r["acc_test"]) / 2
    )
    best_run_idx = best_single_run["run_idx"]
    print(
        f"   A melhor rodada da melhor topologia foi a Rodada {best_run_idx} "
        f"(Treino: {best_single_run['acc_train']:.2%}, Teste: {best_single_run['acc_test']:.2%})"
    )

    plot_confusion_matrix_mean(
        [best_single_run],
        species_names,
        save_path=f"{SAVE_DIR}/best_topo_best_run_confusion_matrix_{timestamp}.png",
        custom_title_prefix="Matriz de Confusão da Melhor Rodada",
        subtitle_suffix=f"Rodada {best_run_idx} da T{best['topology_index']}",
    )

    # 4. Decaimento médio de sigma e learning rate
    plot_decay_comparison(
        best_results,
        save_path=f"{SAVE_DIR}/best_topo_decay_{timestamp}.png",
    )

    # 5. Mapa sinótico da melhor rodada
    best_som = best_single_run["som"]
    plot_synoptic_map(
        best_som,
        species_names,
        save_path=f"{SAVE_DIR}/best_topo_synoptic_map_{timestamp}.png",
        title_suffix=f"Rodada {best_run_idx} da T{best['topology_index']} "
        f"(Grid {best['grid_m']}×{best['grid_n']})",
    )

    # 5b. Matriz de confusão + Mapa sinótico INDIVIDUAL para CADA rodada da melhor topologia
    print(f"\n[GRÁFICOS] Gerando visualizações individuais por rodada ({len(best_results)} rodadas)...")
    for run_result in best_results:
        ridx = run_result["run_idx"]

        # Matriz de confusão individual
        plot_confusion_matrix_mean(
            [run_result],
            species_names,
            save_path=f"{SAVE_DIR}/best_topo_R{ridx}_confusion_matrix_{timestamp}.png",
            custom_title_prefix=f"Matriz de Confusão — Rodada {ridx}",
            subtitle_suffix=(
                f"T{best['topology_index']} R{ridx} "
                f"(Treino: {run_result['acc_train']:.2%}, Teste: {run_result['acc_test']:.2%})"
            ),
        )

        # Mapa sinótico individual
        plot_synoptic_map(
            run_result["som"],
            species_names,
            save_path=f"{SAVE_DIR}/best_topo_R{ridx}_synoptic_map_{timestamp}.png",
            title_suffix=f"T{best['topology_index']} Rodada {ridx} "
            f"(Grid {best['grid_m']}×{best['grid_n']})",
        )

    # 6. Análise de erros da MELHOR RODADA: Ruim classificadas como Ótimo
    print(f"\n{'='*60}")
    print(f"  ANÁLISE DE ERROS — Melhor Rodada (R{best_run_idx})")
    print(f"{'='*60}")

    y_true_raw_best = best_single_run["y_test_raw"]
    y_true_macro_best = best_single_run["y_test_macro"]
    y_pred_macro_best = best_single_run["y_pred_macro_test"]
    y_pred_raw_best = best_single_run["y_pred_raw_test"]

    mask_best = (y_true_macro_best == 2) & (y_pred_macro_best == 0)
    misclass_best = np.where(mask_best)[0]

    best_run_records = []
    if len(misclass_best) == 0:
        print("  ✅ Nenhuma amostra Ruim foi classificada como Ótimo.")
    else:
        print(f"  ⚠️  {len(misclass_best)} amostra(s) Ruim → Ótimo:")
        for idx in misclass_best:
            true_raw = y_true_raw_best[idx]
            pred_raw = y_pred_raw_best[idx]
            print(f"     - Amostra real: {true_raw} → Predita como: {pred_raw}")
            record = {
                "rodada": best_run_idx,
                "acc_teste": f"{best_single_run['acc_test']:.2%}",
                "amostra_real_raw": true_raw,
                "pred_raw": pred_raw,
                "classe_real": "Ruim",
                "classe_pred": "Ótimo",
            }
            # Adicionar colunas de features
            sample_values = best_single_run["X_test"][idx]
            for fname, fval in zip(feature_names, sample_values):
                record[fname] = fval
            best_run_records.append(record)

    analises_dir = "aco_analises"
    os.makedirs(analises_dir, exist_ok=True)

    if best_run_records:
        best_run_misclass_df = pd.DataFrame(best_run_records)
        best_run_csv = f"{analises_dir}/best_run_ruim_como_otimo_{timestamp}.csv"
        best_run_misclass_df.to_csv(best_run_csv, index=False, encoding="utf-8-sig")
        print(f"\n   📄 Detalhes da melhor rodada salvos em: {best_run_csv}")

    # 7. Análise de erros: amostras Ruim classificadas como Ótimo (top 5 rodadas)
    print(f"\n{'='*60}")
    print("  ANÁLISE DE ERROS — Amostras 'Ruim' classificadas como 'Ótimo'")
    print(f"{'='*60}")

    top5_runs = sorted(best_results, key=lambda r: r["acc_test"], reverse=True)[:5]
    all_misclass_records = []

    for rank, run in enumerate(top5_runs, 1):
        run_idx = run["run_idx"]
        y_true_raw = run["y_test_raw"]
        y_true_macro = run["y_test_macro"]
        y_pred_macro = run["y_pred_macro_test"]
        y_pred_raw = run["y_pred_raw_test"]

        # Índices onde verdadeiro=Ruim(2) e predito=Ótimo(0)
        mask = (y_true_macro == 2) & (y_pred_macro == 0)
        misclass_indices = np.where(mask)[0]

        print(f"\n  #{rank} Rodada {run_idx} (Acc Teste: {run['acc_test']:.2%})")

        if len(misclass_indices) == 0:
            print("     ✅ Nenhuma amostra Ruim foi classificada como Ótimo.")
        else:
            print(f"     ⚠️  {len(misclass_indices)} amostra(s) Ruim → Ótimo:")
            for idx in misclass_indices:
                true_raw = y_true_raw[idx]
                pred_raw = y_pred_raw[idx]
                print(f"        - Amostra real: {true_raw} → Predita como: {pred_raw}")
                record = {
                    "rodada": run_idx,
                    "acc_teste": f"{run['acc_test']:.2%}",
                    "amostra_real_raw": true_raw,
                    "pred_raw": pred_raw,
                    "classe_real": "Ruim",
                    "classe_pred": "Ótimo",
                }
                # Adicionar colunas de features
                sample_values = run["X_test"][idx]
                for fname, fval in zip(feature_names, sample_values):
                    record[fname] = fval
                all_misclass_records.append(record)

    # Salvar CSV com os erros
    if all_misclass_records:
        misclass_df = pd.DataFrame(all_misclass_records)
        misclass_csv = f"{analises_dir}/best_topo_ruim_como_otimo_{timestamp}.csv"
        misclass_df.to_csv(misclass_csv, index=False, encoding="utf-8-sig")
        print(f"\n   📄 Detalhes salvos em: {misclass_csv}")
    else:
        print("\n   ✅ Nenhum erro Ruim→Ótimo encontrado nas top 5 rodadas.")

    print("\n✅ Busca de topologias concluída com sucesso!")


if __name__ == "__main__":
    main()
