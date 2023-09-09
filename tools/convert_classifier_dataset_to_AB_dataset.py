import os
import shutil


def read_images(image_dir):
    image_list = []
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_lower = ext.lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg' or ext_lower == '.bmp' or ext_lower == '.tif':
                image_list.append(os.path.join(path, filename))
    return image_list


def copytree(src, dst):
    os.makedirs(dst, exist_ok=True)
    image_list = read_images(src)
    for image_path in image_list:
        shutil.copy2(image_path, os.path.join(dst, os.path.basename(image_path)))


def classifier_to_AB(dataset_dir, ab_dir):
    # We are only considering the train and val datasets.
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    # read label
    label_list = [l for l in sorted(os.listdir(train_dir))]

    # copy
    copytree(os.path.join(train_dir, label_list[0]), os.path.join(ab_dir, 'trainA'))
    copytree(os.path.join(train_dir, label_list[1]), os.path.join(ab_dir, 'trainB'))
    copytree(os.path.join(val_dir, label_list[0]), os.path.join(ab_dir, 'trainA'))
    copytree(os.path.join(val_dir, label_list[1]), os.path.join(ab_dir, 'trainB'))


if __name__ == '__main__':
    dataset_dir = '../dataset/fivesix_bias_dataset_32'
    ab_dir = dataset_dir + '_AB'

    os.makedirs(ab_dir, exist_ok=True)
    classifier_to_AB(dataset_dir, ab_dir)
