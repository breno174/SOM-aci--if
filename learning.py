import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from implement.som import SOM
from src.helper.kmeans import KMeans, SimpleKMeans
from src.plot import plotar_epocas, plotar_taxa_aprendizado, plotar_eqm, plotar_clusters
from data.normalize import Normalize

def main():
    print("carregando dados")
    norm = Normalize("data/Teen_Mental_Health_Dataset.csv")
    data_raw = norm.usage_coluns()
    print("dados carregados")
    print(data_raw)
    data = np.array(data_raw, dtype=float)
    print("data array:\n", data)
    data = np.nan_to_num(data)
    print("data: \n", data)

    # Também carregar stress_level como label ordinal (0=Low, 1=Medium, 2=High)
    df_full = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")
    stress_map = {v: i for i, v in enumerate(sorted(df_full['stress_level'].unique()))}
    stress_int = df_full['stress_level'].map(stress_map).values
    stress_names = [k for k, _ in sorted(stress_map.items(), key=lambda x: x[1])]
    num_classes = len(stress_names)
    print(f"  Classes de stress: {stress_names}")

    # Inicializa SOM
    dim = data.shape[1]
    print(dim)
    origin = np.zeros(dim)
    print(origin)

    som = SOM(line=5, column=5, learning_rate=0.3, dimension=dim, initial_weights=origin, sigma=3)
    som.train(data=data, num_epoch=40, labeled_data=(data, stress_int))

    # # Plota os resultados
    # plotar_epocas(som)
    # plotar_taxa_aprendizado(som)
    # plotar_eqm(som)
    # plotar_clusters(som)

if __name__ == "__main__":
    main()
