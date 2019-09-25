
# Generate function which obtains labels for each image based on presence of specific tag returns a list of with a single tag for each image
ann_dict  = {'kitchen': ['kitchen', 'messy'], 'bathroom': ['bathroom', 'messy'], 'bedroom': ['bedroom', 'messy']}
def annotate_scatter(top5_array, ann_list, ann_dict):
    ann_dict =  None
    ann_list =  None
    ann_out = []
    index_pos = []
    index_count = 0
    add_ann = None
    ann_out_dict = {}
    for img in top5_array:
        score = 2
        add_ann = None
        # Annotate scatter with a list of tags
        if ann_dict is not None:
                for a in ann_dict:
                    if(set(ann_dict[a]).issubset(set(img[1]))):
                        ann_out_dict[img[0]] = ann_dict[a]
                        index_pos.append(index_count)
                        index_count += 1
                return ann_dict, index_pos

        # Annotate scatter with a single tags
        if ann_list is not None:
                for a in ann_list:
                    if a in img[1] and img[1].index(a) < score:
                        print(img[1].index(a))
                        print(img[1])
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