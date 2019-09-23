import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
import scipy








# Generate a list of tags
# possible_tags = pickle.load(open('possible_tags.pkl', 'rb'))

# tags = []
# logging.info('Testing: get embedding of all possible tags')
# for tag in possible_tags:
#     tags.append(tag)



from itertools import zip_longest
import matplotlib.pyplot as plt
import matplotlib

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
# Generate function which obtains labels for each image based on presence of specific tag returns a list of with a single tag for each image
def annotate_scatter(top5_array, ann_list):
    ann_out = []
    index_pos = []
    index_count = 0
    add_ann = None
    for img in top5_array:
        score = 6
        if ann_list[0] in  img[1]:
            add_ann = ann_list[0]
            score = img[1].index(ann_list[0])
        if ann_list[1] in  img[1]:
            if img[1].index(ann_list[1]) < score:
                add_ann = ann_list[1]
                score = img[1].index(ann_list[1])
        if ann_list[2] in  img[1]:
            if img[1].index(ann_list[2]) < score:
                add_ann = ann_list[2]
                score = img[1].index(ann_list[2])
        if ann_list[3] in  img[1]:
            if img[1].index(ann_list[3]) < score:
                add_ann = ann_list[3]
                score = img[1].index(ann_list[3])
        if ann_list[4] in  img[1]:
            if img[1].index(ann_list[4]) < score:
                add_ann = ann_list[4]
                score = img[1].index(ann_list[4])
        # add anootation to list
        if add_ann is not None:
            ann_out.append(add_ann)
            index_pos.append(index_count)
            index_count += 1
        else:
            index_count += 1
    return ann_out, index_pos

f = open('/newvolume/i2t_results.txt', 'r')
# Array of top 5 tags for each image
X = [np.array([line1, line2.replace(" ", "").split(',')], dtype=object) for line1, line2 in grouper(2, f)]

# Generate annotation tag for each image
annot_list, indices_list = annotate_scatter(X, ["kitchen", "bedroom", "bathroom", "washroom", "tarmac"])



# Load score matrix
scores_obj = np.load('/newvolume/score_matrix.npz')
scores = scores_obj['scores']

# Slice out the scores relating to the images tags with the relevant tags
score_subset = list(map(scores.__getitem__, indices_list))

# Generate MDS object
mds = MDS(n_components=2, dissimilarity="precomputed")

# Calculate euclidean distance between each image word vector
similarities = euclidean_distances(score_subset)

pos = mds.fit(similarities).embedding_

fig = plt.figure(figsize=(12,10))

colors = ['red','blue','green','orange', 'black']

plt.scatter(pos[:, 0], pos[:, 1], label= annot_list, cmap=colors)

# colors = {'kitchen':'red', 'bedroom':'blue', 'bathroom':'green', 'washroom':'black', 'tarmac': 'orange', 'notlabelled': 'white'}

# col_list = [c for c in map(lambda x: colors[x],annot_list)]
# plt.scatter(pos[:, 0], pos[:, 1], c= col_list)
plt.show()

plt.savefig('/newvolume/images_2000.pdf')


# ax = plt.subplots(1)
# # Plot and label scatter plot
# for val in range(len(pos)):
#     plt.scatter(pos[val, 0], pos[val, 1], label= annot_list[val])

# ax.legend()

# # plt.scatter(pos[:, 0], pos[:, 1])














# import chart_studio.plotly as py
# from plotly.offline import plot
# import plotly.graph_objs as go

# import numpy as np

# from sklearn import manifold
# from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# n_samples = 20
# seed = np.random.RandomState(seed=3)
# X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
# X_true = X_true.reshape((n_samples, 2))
# # Center the data
# X_true -= X_true.mean()

# similarities = euclidean_distances(X_true)

# # Add noise to the similarities
# noise = np.random.rand(n_samples, n_samples)
# noise = noise + noise.T
# noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
# similarities += noise

# mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=1)
# pos = mds.fit(similarities).embedding_

# print(pos)

# pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
# print(pos)

# # Rotate the data
# clf = PCA(n_components=2)
# X_true = clf.fit_transform(X_true)

# pos = clf.fit_transform(pos)



# fig = plt.figure(figsize=(12,10))

# plt.scatter(pos[:, 0], pos[:, 1])
# plt.scatter(X_true[:, 0], X_true[:, 1])

# plt.show()

# data = []
# p1 = go.Scatter(x=X_true[:, 0], y=X_true[:, 1],
#                 mode='markers+lines',
#                 marker=dict(color='navy', size=10),
#                 line=dict(width=1),
#                 name='True Position')
# data.append(p1)
# p2 = go.Scatter(x=pos[:, 0], y=pos[:, 1],
#                 mode='markers+lines',
#                 marker=dict(color='turquoise', size=10),
#                 line=dict(width=1),
#                 name='MDS')
# data.append(p2)





# layout = go.Layout(xaxis=dict(zeroline=False, showgrid=False,
#                               ticks='', showticklabels=False),
#                    yaxis=dict(zeroline=False, showgrid=False,
#                               ticks='', showticklabels=False),
#                    height=900, hovermode='closest')
# fig = go.Figure(data=data, layout=layout)

# plot(fig)
