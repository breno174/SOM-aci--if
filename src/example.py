import sys
import os
# Adicionando o diretório raiz ao path para que o Python encontre o módulo 'data' e 'src' tranquilamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.som import SOM
from data.normalize import Normalize

def main():
    # 1. Carregar os dados reais filtrados
    # Obs: o caminho do csv leva em conta a execução partindo do diretório raiz
    norm = Normalize("data/Teen_Mental_Health_Dataset.csv")
    data = norm.usage_coluns()
    
    # Forçar conversão para float
    data = np.array(data, dtype=float)
    
    # Normalização min-max (colocar features entre 0 e 1)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    # Evitar divisão por zero se alguma coluna tiver valor constante
    diff = data_max - data_min
    diff[diff == 0] = 1 
    
    data = (data - data_min) / diff
    data = np.nan_to_num(data) # Tratamento segurança

    # 2. Definir o Ponto de Origem (zerado ou centroide)
    dim = data.shape[1]
    origin_point = np.zeros(dim) # Como você pediu, os pesos partirão de um ponto de origem (0, 0, ...)

    # 3. Inicializar SOM
    m, n = 10, 10 # Tamanho da Grade
    print(f"Inicializando SOM {m}x{n} (Dimensão do Neurônio: {dim}).")
    print(f"Ponto de Origem inicial (Pesos): {origin_point}")
    
    som = SOM(line=m, column=n, dim=dim, learning_rate=0.1, initial_weights=origin_point)

    print("Iniciando treinamento da SOM com dataset de Saúde Mental Adolescente...")
    som.train(data, num_epochs=10)
    print("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()
