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
from shutil import move
import sys


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
    feature_type = ["Fireplaces", "Hardwood_Floors", "Kitchen_Islands", "Skylights", "ADE20K_tagged"]
    room_type = ["living_room", "kitchen", "bedroom", "bathroom"]
    # Iterate over images collected for each feature type
    stop = set(nltk.corpus.stopwords.words('english'))
    logging.info('Count word frequencies, number of annotations = %d', len(annotations))
    bar = progressbar.ProgressBar()
    move_file = []
    for feature in feature_type:
        for room in room_type:
            ann_directory = "/newvolume/{}/images/image_annotations/{}/ann".format(feature, room)
            img_directory = "/newvolume/{}/images/images/{}".format(feature, room)
            for filename in os.listdir(ann_directory):
                img_file = filename.split('.json')[0]
                file_path = img_directory + "/" + img_file
                ann_path = ann_directory + "/" + filename
                if filename != ".DS_Store":
                    with open(ann_path) as f:
                        ann = json.load(f)
                        img_count[file_path] = {}
                        img_captions[file_path] = ann['tags']
                        for ob in ann['tags']:
                            if ob['name'] == "Kitchen":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 2 # Increase count for room type to ensure tag is included for each image
                            if ob['name'] == "Living Room":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 2
                            if ob['name'] == "Bedroom":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 2
                            if ob['name'] == "Bathroom":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 2
                            if ob['name'] == "Kitchen Island":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[1]] = 1 # take the second word in order to ID the word island
                            if ob['name'] == "Fireplace":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                            # if ob['name'] == "Skylight":
                            #     tokens = nltk.word_tokenize(ob['name'])
                            #     tokens = [w.lower() for w in tokens]
                            #     tokens = [w for w in tokens if not w in stop]
                            #     img_count[file_path][tokens[0]] = 1
                            if ob['name'] == "Wood Floor":
                                tokens = nltk.word_tokenize(ob['name'])
                                tokens = [w.lower() for w in tokens]
                                tokens = [w for w in tokens if not w in stop]
                                img_count[file_path][tokens[0]] = 1
                        # move annotation and image file to another folder area if image dies not have the appropriate number of tags
                        print(img_count[file_path])
                        print(len(img_count[file_path]))
                        # sys.exit("Error message")
                        if len(img_count[file_path]) < args.tagsPerImage:
                            if not os.path.exists('/newvolume/moved_files'):
                                # make directory to move file to
                                os.mkdir('/newvolume/moved_files')
                            # move image file
                            # generate file paths to allow file to be moved back afterwards
                            move_file.append(file_path)
                            move(file_path, '/newvolume/moved_files')
                            # move annotation file
                            move_file.append(ann_path)
                            move(ann_path, '/newvolume/moved_files')
                            del img_count[file_path]
                            del img_captions[file_path]

    np.savez_compressed('move_file', source_paths = np.array(move_file))


def copy_images_back():
    move_file = np.load('/home/ubuntu/CCA-images-text/main/move_file.npz')
    move_file_paths = move_file['source_paths']

    for path in move_file_paths:
        # get file name
        split_f = path.split('/')[-1]
        source_file_path = '/newvolume/moved_files/' + split_f
        move(source_file_path, path)



def calc_features():
    model = KeyedVectors.load_word2vec_format('/newvolume/text.model.bin', binary=True)
    # Load my own custom room type multilabel classifier
    # net = load_model('/newvolume/resnet_classifier')
    net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    TAGS_PER_IMAGE = args.tagsPerImage
    print ('Tags per image', TAGS_PER_IMAGE)
    img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 4096), dtype=np.float32)
    # img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 256), dtype=np.float32)
    tag_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 200), dtype=np.float32)

    possible_tags = set()

    f = open('train_tags_room_data.txt', 'w')
    pos = 0
    logging.info('Training: calculate image features, choose tag for each image')
    bar = progressbar.ProgressBar()

    for image_id, words in bar(img_count.items()):
        # file_name = coco_train.imgs[image_id]['file_name']
        # img = image.load_img('/newvolume/train2014/' + file_name, target_size=(224, 224))
        file_name = image_id.split('.json')
        img = image.load_img(file_name[0], target_size=(224, 224, 3))

        words_list = []
        words_count = []
        for w in words:
            print(w)
            if w in model.wv.vocab:
                words_list.append(w)
                words_count.append(img_count[image_id][w])

        words_count = np.array(words_count)
        index = np.argsort(words_count)[::-1]
        print(words_list)
        print(words_count)
        # f.write(coco_train.imgs[image_id]['flickr_url'] + '\n')
        f.write(image_id + '\n')

        for i in range(TAGS_PER_IMAGE):
            if i < len(index):
                continue
            else:
                print(image_id)
                print(index)
                print(len(index))
                print(words_list)
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
        # print(features)
        features = features.reshape(-1)

        for i in range(TAGS_PER_IMAGE):
            ind = index[i]
            img_features[TAGS_PER_IMAGE * pos + i,:] = features
            tag_features[TAGS_PER_IMAGE * pos + i,:] = model[ words_list[ind] ]
        possible_tags.add(words_list[ index[0] ])

        pos += 1
        if pos % 20000 == 0:
            logging.info('Training: saving features calculated for the first %d images', pos)
            np.savez_compressed('train_features_room_data', img_features=img_features[:TAGS_PER_IMAGE * pos,:], tag_features=tag_features[:TAGS_PER_IMAGE * pos,:])

    if args.incl_coco == True:
        # load Coco data outputs
        train_features_coco = np.load('train_features.npz')
        img_features_coco = train_features_coco['img_features']
        tag_features_coco = train_features_coco['tag_features']

        # append to img_features data generated from room data
        img_features_joined = np.append(img_features_coco, img_features, axis=0)
        tag_features_joined = np.append(tag_features_coco, tag_features, axis=0)


        logging.info('Training: saving features calculated for all the images')
        np.savez_compressed('train_features_joined', img_features=img_features_joined, tag_features=tag_features_joined)
    else:
        logging.info('Training: saving features calculated for all the images')
        np.savez_compressed('train_features', img_features=img_features, tag_features=tag_features)

    # load tags from Coco data
    possible_tags_coco = pickle.load(open('possible_tags.pkl', 'rb'))
    for t in possible_tags:
        possible_tags_coco.add(t)


    logging.info('Training: number of possible tags = %d', len(possible_tags))
    pickle.dump(possible_tags_coco, open('possible_tags_joined.pkl', 'wb'))


if __name__ == "__main__":
    logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tagsPerImage', default=2, type=int, help='amount of tags per image')
    parser.add_argument('--incl_coco', action='store_true', help='amount of tags per image')
    args = parser.parse_args()

    annFile = '/newvolume/annotations/captions_train2014.json'
    coco_train = COCO(annFile)
    ids = coco_train.getAnnIds()
    annotations = coco_train.loadAnns(ids)

    img_count = {}
    img_captions = {}
    count_words_google_data()

    calc_features()
