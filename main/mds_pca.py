import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances


# # Load score matrix
# scores_obj = np.load('score_matrix.npz')
# scores = scores_obj['scores']

# # Generate MDS object
# mds = MDS(n_components=2, dissimilarity="precomputed")

# # Calculate euclidean distance between each image word vector
# similarities = euclidean_distances(scores)

# mds.fit(similarities)







import chart_studio.plotly as py
from plotly.offline import plot
import plotly.graph_objs as go

import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

n_samples = 20
seed = np.random.RandomState(seed=3)
X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
X_true = X_true.reshape((n_samples, 2))
# Center the data
X_true -= X_true.mean()

similarities = euclidean_distances(X_true)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

print(pos)

pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
print(pos)

# Rotate the data
clf = PCA(n_components=2)
X_true = clf.fit_transform(X_true)

pos = clf.fit_transform(pos)


data = []
p1 = go.Scatter(x=X_true[:, 0], y=X_true[:, 1],
                mode='markers+lines',
                marker=dict(color='navy', size=10),
                line=dict(width=1),
                name='True Position')
data.append(p1)
p2 = go.Scatter(x=pos[:, 0], y=pos[:, 1],
                mode='markers+lines',
                marker=dict(color='turquoise', size=10),
                line=dict(width=1),
                name='MDS')
data.append(p2)

layout = go.Layout(xaxis=dict(zeroline=False, showgrid=False,
                              ticks='', showticklabels=False),
                   yaxis=dict(zeroline=False, showgrid=False,
                              ticks='', showticklabels=False),
                   height=900, hovermode='closest')
fig = go.Figure(data=data, layout=layout)

plot(fig)
