import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
import scipy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from pycocotools.coco import COCO
from annotation_scatter import annotate_scatter
import shutil

def getImage(path):
    return OffsetImage(plt.imread(path, 0), zoom=0.1)





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


f = open('/home/ubuntu/CCA-images-text/main/i2t_results.txt', 'r')
# Array of top 5 tags for each image
X = [np.array([line1, line2.replace(" ", "").split(',')], dtype=object) for line1, line2 in grouper(2, f)]

# Generate annotation tag for each image
# ann_dict  = {'kitchen_counter': ['kitchen', 'counter'], 'kitchen_refrigerator': ['kitchen', 'refrigerator']}
# ann_dict  = {'kitchen_counter': ['kitchen', 'counter']}
# ann_dict  = {'kitchen_refrigerator': ['kitchen', 'refrigerator']}
ann_dict  = {'kitchen': ['kitchen'], 'bedroom': ['bedroom'], 'bathroom': ['bathroom'], 'living': ['living']}
# annot_list, indices_list = annotate_scatter(X, ann_list = ["bathroom"])
annot_list, indices_list = annotate_scatter(X, ann_dict = ann_dict)

# annot_list, indices_list = annotate_scatter(X, ["dog", "cat"])
print(annot_list)
print(len(annot_list))
print(len(indices_list))


def gen_scatter_multi_tag(annot_list, indices_list):
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

    # label_list = ['kitchen counter', 'kitchen refrigerator']
    # label_list = ['kitchen refrigerator']
    label_list = ['kitchen', 'bedroom', 'bathroom', 'living room']

    group = np.array(annot_list)

    # colors = {'kitchen counter':'red', 'kitchen refrigerator': 'blue'}
    colors = {'kitchen':'red', 'bedroom': 'blue', 'bathroom': 'green', 'living': 'orange'}

    col_list = [c for c in map(lambda x: colors[x],annot_list)]

    fig, ax = plt.subplots()

    scatter_x = np.array(pos[:,0])
    scatter_y = np.array(pos[:,1])

################################################################################
# # Uncomment to add coloured dots instead of images to scatter plot #############
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c = colors[g],  label = g)
    ax.legend(loc='lower right')
################################################################################

################################################################################
# Uncomment section below to add images instead of dots as points of scatter plot
    # Plot image instead of point
    # obtain file paths for each image
    annFile = '/newvolume/annotations/instances_val2014.json'
    coco_val = COCO(annFile)
    ids = coco_val.getAnnIds()
    annotations = coco_val.loadAnns(ids)

    img_info = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in img_info:
            img_info[image_id] = coco_val.imgs[image_id]

    img_path_list = []
    for image_id, info in img_info.items():
        file_name = info['file_name']
        img = '/newvolume/val2014/' + file_name
        img_path_list.append(img)

    # Slice out the relevant images
    img_subset = list(map(img_path_list.__getitem__, indices_list))

    print(len(img_subset))
    # dest = '/newvolume/kitchen_counter'
    dest = '/newvolume/mds_results'
    for x0, y0, path in zip(scatter_x, scatter_y,img_subset):
        print(path)
        shutil.copy(path, dest)
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.scatter(pos[:, 0], pos[:, 1], c= col_list)
################################################################################


    plt.show()

    plt.savefig('/newvolume/images_room_type.pdf')

# gen_scatter_multi_tag(annot_list, indices_list)




def gen_scatter_single_tag(annot_list, indices_list):
    # Load score matrix
    scores_obj = np.load('/newvolume/score_matrix.npz')
    scores = scores_obj['scores']

    print(len(scores))

    # Slice out the scores relating to the images tags with the relevant tags
    score_subset = list(map(scores.__getitem__, indices_list))

    # Generate MDS object
    mds = MDS(n_components=2, dissimilarity="precomputed")

    # Calculate euclidean distance between each image word vector
    similarities = euclidean_distances(score_subset)

    pos = mds.fit(similarities).embedding_
    print(len(pos))

    fig = plt.figure(figsize=(12,10))

    # colors = ['red','blue','green','orange', 'black']
    # label_list = ['kitchen', 'bedroom', 'bathroom', 'living room']
    label_list = ['kitchen']
    # label_list = ['bathroom']
    # label_list = ['dog', 'cat']

    group = np.array(annot_list)

    # colors = {'kitchen':'red', 'bedroom':'blue', 'bathroom':'green', 'living':'orange'}
    colors = {'kitchen':'red'}
    # colors = {'dog':'red', 'cat':'blue'}
    col_list = [c for c in map(lambda x: colors[x],annot_list)]
    print(len(col_list))
    print(col_list)
    fig, ax = plt.subplots()

    scatter_x = np.array(pos[:,0])
    scatter_y = np.array(pos[:,1])
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c = colors[g],  label = g)

################################################################################
# Uncomment section below to add images instead of dots as points of scatter plot
    # Plot image instead of point
    # obtaine file paths for each image
    # annFile = '/newvolume/annotations/instances_val2014.json'
    # coco_val = COCO(annFile)
    # ids = coco_val.getAnnIds()
    # annotations = coco_val.loadAnns(ids)

    # img_info = {}
    # for ann in annotations:
    #     image_id = ann['image_id']
    #     if image_id not in img_info:
    #         img_info[image_id] = coco_val.imgs[image_id]

    # img_path_list = []
    # for image_id, info in img_info.items():
    #     file_name = info['file_name']
    #     img = '/newvolume/val2014/' + file_name
    #     img_path_list.append(img)

    # # Slice out the relevant images
    # img_subset = list(map(img_path_list.__getitem__, indices_list))


    # dest = '/newvolume/bathroom'
    # for x0, y0, path in zip(scatter_x, scatter_y,img_subset):
    #     print(path)
    #     # shutil.copy(path, dest)
    #     ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    #     ax.add_artist(ab)
    ################################################################################

    ax.legend(loc='lower right')
    # colors = {'kitchen':'red', 'bedroom':'blue', 'bathroom':'green', 'washroom':'black', 'tarmac': 'orange', 'notlabelled': 'white'}

    # col_list = [c for c in map(lambda x: colors[x],annot_list)]
    # plt.scatter(pos[:, 0], pos[:, 1], c= col_list)

    # col_list = [c for c in map(lambda x: colors[x],annot_list)]
    # plt.scatter(pos[:, 0], pos[:, 1], c= col_list)
    plt.show()

    # plt.savefig('/newvolume/images_room_type.pdf')
    plt.savefig('/newvolume/images_kitchen.pdf')

gen_scatter_single_tag(annot_list, indices_list)























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
