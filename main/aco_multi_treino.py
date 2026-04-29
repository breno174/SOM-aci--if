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
from main.aco_analises import (
    train_test_split_hierarchical,
    convert_raw_to_macro,
)

# ── Configurações ──────────────────────────────────────────────────
NUM_EPOCHS = 140  # épocas de treinamento
GRID_M = 7  # linhas do grid SOM
GRID_N = 7  # colunas do grid SOM
LR = 0.3  # taxa de aprendizado inicial
SIGMA = 7  # raio de vizinhança inicial
N_CLUSTERS = 5  # número de clusters K-Means (3 macro-classes: Ótimo, Normal, Ruim)
NUM_RUNS = 10  # número de treinos independentes
K_BMU = 5  # número de BMUs para predição
SAVE_DIR = "src/acoPictures/multi"  # diretório para salvar gráficos
EXPERIMENT_DIR = "src/experiments/multi"  # diretório para salvar experimentos


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
    cols_to_drop = [5, 12, 13]
    X = df.drop(df.columns[cols_to_drop + [-1]], axis=1).values.astype(float)
    print(f"   {X.shape[0]} amostras | dim={X.shape[1]}")

    # Normalizando os dados (Z-score)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, species_int, y_raw.values, species_names


def run_single_training(X, species_int, y_raw_values, seed):
    """Executa um treino completo do SOM e retorna métricas."""
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
    initial_weights_snapshot = som.weights.copy()

    # 1. separar dados com seed diferente para cada rodada
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = (
        train_test_split_hierarchical(
            X, species_int, y_raw_values, test_size=0.2, seed=seed
        )
    )

    # 2. treinar SOM (sem imprimir épocas individuais)
    som.train(
        X_train, num_epoch=NUM_EPOCHS, X_test=X_test, y_train=y_train, y_test=y_test
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
    }


def plot_accuracy_comparison(all_results, species_names, save_path):
    """
    Gráfico 1: Acurácia média de treino vs teste por rodada,
    com linha de média geral.
    """
    n_runs = len(all_results)
    acc_trains = [r["acc_train"] for r in all_results]
    acc_tests = [r["acc_test"] for r in all_results]

    x = np.arange(1, n_runs + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        [a * 100 for a in acc_trains],
        color="#2196F3",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Acurácia Treino",
    )
    ax.plot(
        x,
        [a * 100 for a in acc_tests],
        color="#F44336",
        linewidth=2,
        marker="s",
        markersize=6,
        label="Acurácia Teste",
    )

    mean_train = np.mean(acc_trains) * 100
    mean_test = np.mean(acc_tests) * 100

    ax.axhline(
        y=mean_train,
        color="#2196F3",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Média Treino ({mean_train:.2f}%)",
    )
    ax.axhline(
        y=mean_test,
        color="#F44336",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Média Teste ({mean_test:.2f}%)",
    )

    ax.set_title(
        f"Acurácia por Rodada — {n_runs} Treinos ({K_BMU}-BMU)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Rodada", fontsize=11)
    ax.set_ylabel("Acurácia (%)", fontsize=11)
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de acurácia salvo em: {save_path}")
    plt.close(fig)


def plot_mse_comparison(all_results, save_path):
    """
    Gráfico 2: Dois subplots lado a lado.
    - Esquerda: EQM médio por época de TREINO (média de todas as rodadas).
    - Direita: EQM médio por época de TESTE (média de todas as rodadas).
    """
    n_runs = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(1, len(all_results[0]["mse_train_history"]) + 1))

    # ── Esquerda: EQM médio de TREINO ──
    all_mse_train = np.array([r["mse_train_history"] for r in all_results])
    mean_train = np.mean(all_mse_train, axis=0)
    std_train = np.std(all_mse_train, axis=0)

    axes[0].plot(
        epochs,
        mean_train,
        color="#2196F3",
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
        color="#2196F3",
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
    all_mse_test = np.array([r["mse_test_history"] for r in all_results])
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


def plot_confusion_matrix_mean(all_results, species_names, save_path):
    """
    Gráfico 3: Duas matrizes de confusão acumuladas lado a lado.
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
        (axes[0], total_train, "Matriz de Confusão Acumulada — TREINO", acc_train),
        (axes[1], total_test, "Matriz de Confusão Acumulada — TESTE", acc_test),
    ]:
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(species_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(species_names, fontsize=10)
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        ax.set_title(
            f"{title}\n{n_runs} rodadas (Acc={acc:.2%})",
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

    all_sigma = np.array([r["history_sigma"] for r in all_results])
    all_lr = np.array([r["history_lr"] for r in all_results])

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


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carregar dados uma vez
    X, species_int, y_raw_values, species_names = load_and_prepare_data()

    # ── Múltiplos treinos ──────────────────────────────────────────
    all_results = []

    for run in range(NUM_RUNS):
        seed = 42 + run  # seed diferente para cada rodada
        print(f"\n{'='*60}")
        print(f"  RODADA {run + 1}/{NUM_RUNS}  (seed={seed})")
        print(f"{'='*60}")

        result = run_single_training(X, species_int, y_raw_values, seed=seed)

        print(f"   Acc Treino: {result['acc_train']:.2%}")
        print(f"   Acc Teste:  {result['acc_test']:.2%}")
        print(f"   EQM Final (treino): {result['mse_train_history'][-1]:.4f}")
        print(f"   EQM Final (teste):  {result['mse_test_history'][-1]:.4f}")

        # Salvar experimento da rodada
        experiment_path = f"{EXPERIMENT_DIR}/aco_multi_run{run+1}_{timestamp}.json"
        SOM.save_experiment(
            path=experiment_path,
            seed=seed,
            initial_weights=result["initial_weights"],
            sample_order=[],
            config={
                "grid": [GRID_M, GRID_N],
                "dim": X.shape[1],
                "learning_rate": LR,
                "sigma": SIGMA,
                "num_epochs": NUM_EPOCHS,
                "num_samples": result["num_train_samples"],
                "k_bmu": K_BMU,
                "run": run + 1,
                "acc_train": float(result["acc_train"]),
                "acc_test": float(result["acc_test"]),
            },
        )

        all_results.append(result)

    # ── Resumo geral ──────────────────────────────────────────────
    acc_trains = [r["acc_train"] for r in all_results]
    acc_tests = [r["acc_test"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  RESUMO — {NUM_RUNS} TREINOS")
    print(f"{'='*60}")
    print(
        f"   Acurácia Treino:  {np.mean(acc_trains)*100:.2f}% ± {np.std(acc_trains)*100:.2f}%"
    )
    print(
        f"   Acurácia Teste:   {np.mean(acc_tests)*100:.2f}% ± {np.std(acc_tests)*100:.2f}%"
    )
    print(
        f"   Melhor Teste:     {np.max(acc_tests)*100:.2f}%  (rodada {np.argmax(acc_tests)+1})"
    )
    print(
        f"   Pior Teste:       {np.min(acc_tests)*100:.2f}%  (rodada {np.argmin(acc_tests)+1})"
    )

    # ── Gerar gráficos ──────────────────────────────────────────
    print("\n[GRÁFICOS] Gerando visualizações agregadas...")

    # 1. Acurácia treino vs teste por rodada
    plot_accuracy_comparison(
        all_results,
        species_names,
        save_path=f"{SAVE_DIR}/multi_accuracy_{timestamp}.png",
    )

    # 2. EQM: exemplo + média
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

    print("\n✅ Todos os gráficos foram salvos com sucesso!")


if __name__ == "__main__":
    main()
