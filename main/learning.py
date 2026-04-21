import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from implement.som import SOM
from src.helper.kmeans import KMeans
from src.helper.plot import plot_progression, plot_cluster_grid, plot_final_clusters
from data.normalize import Normalize

# ── Configurações ──────────────────────────────────────────────────
NUM_EPOCHS   = 100          # épocas de treinamento
GRID_M       = 5            # linhas do grid SOM
GRID_N       = 5            # colunas do grid SOM
LR           = 0.3          # taxa de aprendizado inicial
SIGMA        = 3            # raio de vizinhança inicial
N_CLUSTERS   = 5            # número de clusters K-Means
SAVE_DIR     = "src/pictures"   # diretório para salvar gráficos


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Carregar e normalizar dados ──────────────────────────────
    print("[1/7] Carregando e normalizando dados...")
    norm = Normalize("data/Teen_Mental_Health_Dataset.csv")
    data_raw = norm.usage_coluns()
    print("dados carregados")
    print(data_raw)
    data = np.array(data_raw, dtype=float)
    print("data array:\n", data)
    data = np.nan_to_num(data)

    # Normalização Min-Max para [0, 1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    diff = data_max - data_min
    diff[diff == 0] = 1
    data = (data - data_min) / diff

    print(f"      {data.shape[0]} amostras | dim={data.shape[1]}")

    # ── 2. Carregar labels de stress ────────────────────────────────
    df_full = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")
    stress_map = {v: i for i, v in enumerate(sorted(df_full['stress_level'].unique()))}
    stress_int = df_full['stress_level'].map(stress_map).values
    stress_names = [k for k, _ in sorted(stress_map.items(), key=lambda x: x[1])]
    num_classes = len(stress_names)
    print(f"      Classes de stress: {stress_names}")

    # ── 3. Inicializar e treinar SOM ────────────────────────────────
    print(f"[2/7] Treinando SOM ({GRID_M}x{GRID_N}, {NUM_EPOCHS} épocas)...")
    dim = data.shape[1]
    print(dim)
    origin = np.zeros(dim)
    print(origin)

    som = SOM(
        line=GRID_M, column=GRID_N,
        learning_rate=LR, dimension=dim,
        initial_weights=origin, sigma=SIGMA
    )
    som.train(data=data, num_epoch=NUM_EPOCHS, labeled_data=(data, stress_int))
    print("      Treinamento concluído!")

    # ── 4. Progressão dos neurônios por épocas ──────────────────────
    print("[3/7] Plotando evolução dos neurônios por épocas...")
    # Selecionar ~12 snapshots espaçados ao longo de 100 épocas
    # (época 0 = estado após 1ª época, índice 0 no history)
    total = len(som.history)
    indices_to_plot = sorted(set(
        [0] +                                              # início
        list(range(9, total, 10)) +                        # a cada 10 épocas: 10, 20, ...
        [total - 1]                                        # final
    ))
    print(f"      Épocas selecionadas: {[i+1 for i in indices_to_plot]}")
    plot_progression(som.history, data, som, indices_to_plot)

    # ── 5. Curvas de EQM e Acurácia ─────────────────────────────────
    print("[4/7] Plotando EQM e Acurácia...")

    # EQM por época
    fig, ax = plt.subplots(figsize=(10, 4))
    epocas = list(range(1, len(som.history_mse) + 1))
    ax.plot(epocas, som.history_mse, color='crimson', linewidth=2, marker='o', markersize=3, label='EQM')
    ax.fill_between(epocas, som.history_mse, alpha=0.12, color='crimson')
    ax.set_title("Erro Quadrático Médio (EQM) por Época", fontsize=13, fontweight='bold')
    ax.set_xlabel("Época")
    ax.set_ylabel("EQM médio")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mental_eqm.png", dpi=120, bbox_inches='tight')
    print(f"      → Salvo: {SAVE_DIR}/mental_eqm.png")
    plt.show()

    # Acurácia por época
    if som.history_accuracy:
        fig, ax = plt.subplots(figsize=(10, 4))
        acc_pct = [a * 100 for a in som.history_accuracy]
        ax.plot(epocas, acc_pct, color='steelblue', linewidth=2, marker='s', markersize=3, label='Acurácia (%)')
        ax.fill_between(epocas, acc_pct, alpha=0.12, color='steelblue')
        ax.set_title("Acurácia por Época", fontsize=13, fontweight='bold')
        ax.set_xlabel("Época")
        ax.set_ylabel("Acurácia (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/mental_accuracy.png", dpi=120, bbox_inches='tight')
        print(f"      → Salvo: {SAVE_DIR}/mental_accuracy.png")
        plt.show()

    # ── 6. Decaimento do LR e Sigma ─────────────────────────────────
    print("[5/7] Plotando decaimento dos parâmetros...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Decaimento dos Parâmetros da SOM ao Longo das Épocas", fontsize=13, fontweight='bold')

    axes[0].plot(epocas, som.history_sigma, color='darkorange', linewidth=2, marker='o', markersize=2)
    axes[0].fill_between(epocas, som.history_sigma, alpha=0.15, color='darkorange')
    axes[0].set_title("Raio de Vizinhança (σ)", fontweight='bold')
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("σ")
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(epocas, som.history_lr, color='purple', linewidth=2, marker='o', markersize=2)
    axes[1].fill_between(epocas, som.history_lr, alpha=0.15, color='purple')
    axes[1].set_title("Taxa de Aprendizado (α)", fontweight='bold')
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("α")
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mental_neighborhood_decay.png", dpi=120, bbox_inches='tight')
    print(f"      → Salvo: {SAVE_DIR}/mental_neighborhood_decay.png")
    plt.show()

    # ── 7. Clusters K-Means sobre neurônios finais ──────────────────
    print("[6/7] Clustering K-Means sobre pesos finais...")
    kmeans = KMeans(k=N_CLUSTERS)
    kmeans.fit(som.weights)
    labels = kmeans.labels

    plot_final_clusters(som, data, kmeans, labels, stress_int, stress_names)

    # ── 8. Grade de clusters e labels ───────────────────────────────
    print("[7/7] Plotando grade de clusters e labels...")
    plot_cluster_grid(som, labels, stress_int, data, stress_names)

    print("\n✅ Todos os gráficos foram gerados com sucesso!")


if __name__ == "__main__":
    main()

