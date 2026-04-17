import sys
import os
import numpy as np

# Adjust the python path to import correctly
sys.path.append(os.path.abspath('.'))

from src.som import SOM
from src.helper.kmeans import KMeans
from data.normalize import Normalize
import pandas as pd

def run():
    print("Loading data...")
    df_full = pd.read_csv("data/Iris.csv")
    raw_data = np.array(df_full[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], dtype=float)
    raw_min = raw_data.min(axis=0)
    raw_max = raw_data.max(axis=0)
    raw_diff = raw_max - raw_min
    raw_diff[raw_diff == 0] = 1
    data = (raw_data - raw_min) / raw_diff

    dim = data.shape[1]
    m, n = 3, 2
    origin = np.zeros(dim)
    som = SOM(line=m, column=n, dim=dim, learning_rate=0.1, sigma=2, initial_weights=origin)
    
    num_epochs = 8
    num_samples = data.shape[0]
    max_iterations = num_epochs * num_samples
    global_iter = 0

    print("Training SOM for 8 epochs without tracking history to speed up...")
    np.random.seed(42)
    for epoch in range(num_epochs):
        shuffled_indices = np.random.permutation(num_samples)
        data_shuffled = data[shuffled_indices]
        for x in data_shuffled:
            bmu = som.find_bmu(x)
            curr_lr = som._decay(som.learning_rate, global_iter, max_iterations)
            curr_sigma = som._decay(som.sigma, global_iter, max_iterations)
            som._update_weights(x, bmu, curr_lr, curr_sigma)
            global_iter += 1

    final_weights = np.array([node.neuron.weights.copy() for node in som.nodes])
    print("Final Weights:", final_weights)

    kmeans = KMeans(k=3)
    labels = kmeans.fit_predict(final_weights)
    print("KMeans labels:", labels)

if __name__ == '__main__':
    run()
