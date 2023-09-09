import os
import random
from shutil import move

from PIL import Image
from torchvision import datasets
from tqdm import tqdm

val_rate = 0.1
seed = 123


def create_classifier_dataset(mnist_dataset, svhn_dataset, result_dir, label_list):
    for l in label_list[0] + label_list[1]:
        os.makedirs(os.path.join(result_dir, 'train', str(l)), exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'test', str(l)), exist_ok=True)

    c = 0

    for img, label in tqdm(svhn_dataset):
        if label in label_list[0]:
            img.save(os.path.join(result_dir, 'train', str(label), f'{c}.png'))
        elif label in label_list[1]:
            img.save(os.path.join(result_dir, 'test', str(label), f'{c}.png'))
        c += 1

    for img, label in tqdm(mnist_dataset):
        if label in label_list[0]:
            img.save(os.path.join(result_dir, 'test', str(label), f'{c}.png'))
        elif label in label_list[1]:
            img.save(os.path.join(result_dir, 'train', str(label), f'{c}.png'))
        c += 1

    # create val
    for label in tqdm(label_list[0] + label_list[1]):
        train_dir = os.path.join(result_dir, 'train', str(label))
        val_dir = os.path.join(result_dir, 'val', str(label))
        os.makedirs(val_dir, exist_ok=True)

        train_image_list = [os.path.join(train_dir, filename) for filename in sorted(os.listdir(train_dir))]
        random.Random(seed).shuffle(train_image_list)

        val_image_list = train_image_list[:int(len(train_image_list) * val_rate)]
        for image_path in val_image_list:
            move(image_path, image_path.replace(train_dir, val_dir))


def read_mnistm(dataset_dir):
    samples = []

    image_dir = os.path.join(dataset_dir, 'mnist_m_train')
    txt_path = os.path.join(dataset_dir, 'mnist_m_train_labels.txt')

    # read txt
    filename_dict = {}
    with open(txt_path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line_split = line.strip().split(' ')
            filename_dict[line_split[0]] = int(line_split[1])

    for filename, label in filename_dict.items():
        img = Image.open(os.path.join(image_dir, filename))
        samples.append((img, label))

    return samples


if __name__ == '__main__':
    mnist_root = './mnist-data'
    mnist_dataset = datasets.MNIST(root=mnist_root, train=True, download=True, transform=None)

    mnistm_root = './mnist_m'
    svhn_dataset = read_mnistm(mnistm_root)

    label_list = [[5], [6]]

    result_dir = '../dataset/fivesix_bias_dataset'
    create_classifier_dataset(mnist_dataset, svhn_dataset, result_dir, label_list)
