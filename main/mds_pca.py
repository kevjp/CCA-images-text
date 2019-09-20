import numpy as np
from sklearn.manifold import MDS


# Load score matrix
scores_obj = np.load('score_matrix.npz')
scores = scores_obj['scores']

# Generate MDS object
mds = MDS(n_components=2, dissimilarity="precomputed")

mds.fit(scores)