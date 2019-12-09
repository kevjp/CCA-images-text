from sklearn.model_selection import train_test_split
import progressbar
import numpy as np

train_features_coco = np.load('/Users/kevinryan/Documents/DataScienceMSc/Rightmove/CCA_images_text/outputs_COCO_GoogleplusADE20K_VGG16_/train_features_joined.npz')
img_features_coco = train_features_coco['img_features']
tag_features_coco = train_features_coco['tag_features']

print(img_features_coco.shape)
print(tag_features_coco.shape)
(trainX, testX, trainY, testY) = train_test_split(img_features_coco,
    tag_features_coco, test_size=0.2, random_state=42)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape   )
# feature_type = ["Fireplaces", "Hardwood_Floors", "Kitchen_Islands", "Skylights", "ADE20K_tagged"]
# room_type = ["living_room", "kitchen", "bedroom", "bathroom"]
# bar = progressbar.ProgressBar()
# move_file = []
# for feature in feature_type:
#     for room in room_type:
#         ann_directory = "/newvolume/{}/images/image_annotations/{}/ann".format(feature, room)
#         img_directory = "/newvolume/{}/images/images/{}".format(feature, room)
#         for filename in os.listdir(ann_directory):
#             img_file = filename.split('.json')[0]
#             file_path = img_directory + "/" + img_file
#             ann_path = ann_directory + "/" + filename
#             if filename != ".DS_Store":