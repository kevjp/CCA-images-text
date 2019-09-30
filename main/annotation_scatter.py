from collections import OrderedDict

# Generate function which obtains labels for each image based on presence of specific tag returns a list of with a single tag for each image
ann_dict  = {'kitchen': ['kitchen', 'messy'], 'bathroom': ['bathroom', 'messy'], 'bedroom': ['bedroom', 'messy']}
def annotate_scatter(top5_array, ann_list=None, ann_dict=None):
    # variables relating to subset
    ann_out = []
    index_pos = []
    index_count = 0
    add_ann = None
    ann_out_dict = OrderedDict()
    # variables relating to superset key
    ann_key = []
    index_pos_key = []
    index_count_key = 0
    add_ann_key = None
    ann_out_key_dict = OrderedDict()

    # Annotate scatter with a list of tags
    if ann_dict is not None:
        for img in top5_array:
            score_vec = 0
            score_key = 0
            ind_ex = [0]
            ind_ex_key = 0
            add_ann = None
            for a in ann_dict:
                if(set(ann_dict[a]).issubset(set(img[1]))):
                    add_ann = ann_dict[a]
                    ind_ex = [img[1].index(tag) for tag in ann_dict[a]]
                elif a in set(img[1]):
                    # generate points for superset relating to key
                    add_ann_key = a
                    print(img[1].index(a))
                    ind_ex_key = img[1].index(a)
                else:
                    continue
            if sum(ind_ex) > score_vec:
                score_vec = sum(ind_ex)
                ann_out_dict[img[0]] = add_ann
                index_pos.append(index_count)
                index_count += 1
            else:
                index_count += 1
            if ind_ex_key > score_key:
                score_key = ind_ex_key
                ann_out_key_dict[img[0]] = add_ann_key
                index_pos_key.append(index_count_key)
                index_count_key += 1
            else:
                index_count_key += 1

        ann_out = [" ".join(ann_out_dict[elem]) for elem in ann_out_dict]
        ann_key = [elem for elem in ann_out_key_dict]
        return ann_out, index_pos, ann_key, index_pos_key

    # Annotate scatter with a single tags
    if ann_list is not None:
        for img in top5_array:
            score = 2
            add_ann = None
            for a in ann_list:
                if a in img[1] and img[1].index(a) < score:
                    add_ann = a
                    score = img[1].index(a)
            # add anootation to list
            if add_ann is not None:
                ann_out.append(add_ann)
                index_pos.append(index_count)
                index_count += 1
            else:
                index_count += 1
        return ann_out, index_pos