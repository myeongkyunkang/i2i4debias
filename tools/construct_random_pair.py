import os
import random


def read_images(image_dir):
    image_list = []
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_lower = ext.lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg' or ext_lower == '.bmp' or ext_lower == '.tif':
                image_list.append(os.path.join(path, filename))
    return sorted(image_list)


def construct_random_pair(data_dir, result_dir, label_list, seed, trainval=False):
    os.makedirs(result_dir, exist_ok=True)

    if trainval:
        path_list = read_images(os.path.join(data_dir, 'trainval'))
    else:
        train_images = read_images(os.path.join(data_dir, 'train'))
        val_images = read_images(os.path.join(data_dir, 'val'))
        path_list = train_images + val_images

    paths_list = []
    for labels in label_list:
        paths_list.append(list())
        for image_path in path_list:
            if any([l in image_path for l in labels]):
                paths_list[-1].append(image_path)

    s = '' if seed == 0 else seed

    with open(os.path.join(result_dir, f'match{s}.csv'), 'wt') as wf:
        for label, paths in zip(label_list, paths_list):
            for p1 in paths:
                for label_B_index, label_B in enumerate(label_list):
                    if label != label_B:
                        p2 = random.choice(paths_list[label_B_index])
                        wf.write(f'{p1.replace(data_dir, "")},{p2.replace(data_dir, "")}\n')


if __name__ == '__main__':
    seed = 0
    random.seed(seed)

    data_dir = '../dataset/fivesix_bias_dataset'
    result_dir = 'fivesix'
    label_list = [
        ['/5/'],
        ['/6/']
    ]
    construct_random_pair(data_dir, result_dir, label_list, seed)
