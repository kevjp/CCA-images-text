from gensim.models import word2vec, KeyedVectors
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from pycocotools.coco import COCO
import argparse
import logging
import os
import pickle
import progressbar
import nltk
import numpy as np
import json

"""
Need to perform the following copies
scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Fireplaces.zip ubuntu@ec2-18-130-16-124.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Hardwood_Floors.zip ubuntu@ec2-18-130-16-124.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Kitchen_Islands.zip ubuntu@ec2-18-130-16-124.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Skylights.zip ubuntu@ec2-18-130-16-124.eu-west-2.compute.amazonaws.com:/newvolume

cd /newvolume
unzip -q Fireplaces.zip
unzip -q Hardwood_Floors.zip
unzip -q Kitchen_Islands.zip
unzip -q Skylights.zip


"""

def count_words():
    stop = set(nltk.corpus.stopwords.words('english'))
    logging.info('Count word frequencies, number of annotations = %d', len(annotations))
    bar = progressbar.ProgressBar()
    for ann in bar(annotations):
        caption = ann['caption']
        image_id = ann['image_id']
        tokens = nltk.word_tokenize(caption)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if image_id not in img_count:
            img_count[image_id] = {}
            img_captions[image_id] = [caption]
        else:
            img_captions[image_id].append(caption)

        for w in tokens:
            if w in img_count[image_id]:
                img_count[image_id][w] += 1
            else:
                img_count[image_id][w] = 1

    logging.info('Training: number of images = %d', len(img_count))
def count_words_google_data():
    feature_type = ["Fireplaces", "Hardwood_Floors", "Kitchen_Islands", "Skylights"]
    room_type = ["living_room", "kitchen", "bedroom", "bathroom"]
    # Iterate over images collected for each feature type
    stop = set(nltk.corpus.stopwords.words('english'))
    logging.info('Count word frequencies, number of annotations = %d', len(annotations))
    bar = progressbar.ProgressBar()
    for feature in feature_type:
        for room in room_type:
            ann_directory = "/newvolume/{}/images/image_annotations/{}/ann".format(feature, room)
            img_directory = "/newvolume/{}/images/images/{}".format(feature, room)
            for filename in os.listdir(ann_directory):
                img_file = filename.split('.json')[0]
                file_path = img_directory + "/" + img_file
                if filename != ".DS_Store":
                    with open(ann_directory + "/" + filename) as f:
                        ann = json.load(f)
                        img_count[file_path] = {}
                        img_captions[file_path] = ann['tags']
                        for ob in ann['tags']:
                            if ob['name'] == "Kitchen":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Living Room":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Bedroom":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Bathroom":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Kitchen Island":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Fireplace":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Skylight":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Wood Floor":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1



def calc_features():
    model = KeyedVectors.load_word2vec_format('/newvolume/outputs/text.model.bin', binary=True)
    # Load my own custom room type multilabel classifier
    net = load_model('/newvolume/resnet_classifier')
    # net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    TAGS_PER_IMAGE = args.tagsPerImage
    print ('Tags per image', TAGS_PER_IMAGE)
    # img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 4096), dtype=np.float32)
    img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 256), dtype=np.float32)
    tag_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 200), dtype=np.float32)

    possible_tags = set()

    f = open('train_tags.txt', 'w')
    pos = 0
    logging.info('Training: calculate image features, choose tag for each image')
    bar = progressbar.ProgressBar()

    for image_id, words in bar(img_count.items()):
        # file_name = coco_train.imgs[image_id]['file_name']
        # img = image.load_img('/newvolume/train2014/' + file_name, target_size=(224, 224))
        file_name = image_id.split('.json')
        img = image.load_img(file_name[0], target_size=(224, 224))

        words_list = []
        words_count = []
        for w in words:
            print(w)
            if w in model.wv.vocab:
                words_list.append(w)
                words_count.append(img_count[image_id][w])

        words_count = np.array(words_count)
        index = np.argsort(words_count)[::-1]

        # f.write(coco_train.imgs[image_id]['flickr_url'] + '\n')
        f.write(image_id + '\n')
        for i in range(TAGS_PER_IMAGE):
            f.write(words_list[ index[i] ] + '\n')
        for i in range(0,min(5,len(index))):
            ind = index[i]
            f.write(words_list[ind] + ', ' + str(words_count[ind]) + '\n')
        for caption in img_captions[image_id]:
            f.write(caption['name'] + '\n')

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = features.reshape(-1)

        for i in range(TAGS_PER_IMAGE):
            ind = index[i]
            img_features[TAGS_PER_IMAGE * pos + i,:] = features
            tag_features[TAGS_PER_IMAGE * pos + i,:] = model[ words_list[ind] ]
        possible_tags.add(words_list[ index[0] ])

        pos += 1
        if pos % 20000 == 0:
            logging.info('Training: saving features calculated for the first %d images', pos)
            np.savez_compressed('train_features', img_features=img_features[:TAGS_PER_IMAGE * pos,:], tag_features=tag_features[:TAGS_PER_IMAGE * pos,:])

    logging.info('Training: saving features calculated for all the images')
    np.savez_compressed('train_features', img_features=img_features, tag_features=tag_features)

    logging.info('Training: number of possible tags = %d', len(possible_tags))
    pickle.dump(possible_tags, open('possible_tags.pkl', 'wb'))


if __name__ == "__main__":
    logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tagsPerImage', default=2, type=int, help='amount of tags per image')
    args = parser.parse_args()

    annFile = '/newvolume/annotations/captions_train2014.json'
    coco_train = COCO(annFile)
    ids = coco_train.getAnnIds()
    annotations = coco_train.loadAnns(ids)

    img_count = {}
    img_captions = {}
    count_words_google_data()

    calc_features()
