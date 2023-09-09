import os

from PIL import Image
from tqdm import tqdm


def resize_with_padding(im, desired_size):
    old_size = im.size
    if old_size[0] == desired_size and old_size[1] == desired_size:
        return im
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    if old_size[0] != new_size[0] or old_size[1] != new_size[1]:
        im = im.resize(new_size, Image.ANTIALIAS)
    if new_size[0] == desired_size and new_size[1] == desired_size:
        return im
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def read_images(image_dir):
    image_list = []
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_lower = ext.lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg' or ext_lower == '.bmp' or ext_lower == '.tif':
                image_list.append(os.path.join(path, filename))
    return image_list


if __name__ == '__main__':
    image_size = 32
    input_dir = '../dataset/fivesix_bias_dataset'
    result_dir = f'../dataset/fivesix_bias_dataset_{image_size}'

    image_list = read_images(input_dir)
    for image_path in tqdm(image_list):
        img = Image.open(image_path)
        resized_img = resize_with_padding(img, desired_size=image_size)

        # save
        save_path = image_path.replace(input_dir, result_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        resized_img.save(save_path)
