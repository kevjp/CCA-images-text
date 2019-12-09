import os

feature_type = ["Fireplaces", "Hardwood_Floors", "Kitchen_Islands", "Skylights", "ADE20K_tagged"]
room_type = ["living_room", "kitchen", "bedroom", "bathroom"]


# Iterate over images collected for each feature type
tag_list = {}
for feature in feature_type:
    for room in room_type:
        ann_directory = "/Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Hardwood_Floors/images/image_annotations/bathroom/ann".format(feature, room)
        for filename in os.listdir(ann_directory):
            if filename != ".DS_Store":
                with open(ann_directory + "/" + filename) as f:
                    ann = json.load(f)
                    tag_list[filename] = {}
                    for ob in ann['tags']:
                        if ob['name'] == "Kitchen":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Living Room":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Bedroom":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Bathroom":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Kitchen Island":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Fireplace":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Skylight":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1
                        if ob['name'] == "Wood Floor":
                            tokens = nltk.word_tokenize(ob['name'])
                            tokens = [w.lower() for w in tokens]
                            tokens = [w for w in tokens if not w in stop]
                            tag_list[filename][tokens[0]] = 1