import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Parameters
L = 20  # Lattice size
T_range = np.linspace(1.0, 3.5, 20)  # Temperature range
n_steps = 1000  # Monte Carlo steps per temperature

# Initialize a random Ising lattice
lattice = np.random.choice([-1, 1], size=(L, L))

def metropolis_step(lattice, T):
    L = lattice.shape[0]
    for _ in range(L**2):
        i, j = np.random.randint(0, L, size=2)
        delta_E = 2 * lattice[i, j] * (
            lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + 
            lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
        )
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            lattice[i, j] *= -1

def generate_data():
    data, labels = [], []
    for T in T_range:
        lattice = np.random.choice([-1, 1], size=(L, L))
        for _ in range(n_steps):
            metropolis_step(lattice, T)
        data.append(lattice.ravel())  # Flatten lattice for ML
        labels.append(T)
    return np.array(data), np.array(labels)

# Generate Monte Carlo data
data, labels = generate_data()

# Apply PCA
def apply_pca(data):
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)
    return transformed_data

pca_result = apply_pca(data)

# Apply t-SNE
def apply_tsne(data):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    transformed_data = tsne.fit_transform(data)
    return transformed_data

tsne_result = apply_tsne(data)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm')
plt.colorbar(sc, ax=axes[0], label='Temperature')
axes[0].set_title('PCA')
sc = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='coolwarm')
plt.colorbar(sc, ax=axes[1], label='Temperature')
axes[1].set_title('t-SNE')
plt.show()
