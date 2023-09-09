import os

from PIL import Image
from tqdm import tqdm

import models
import util
from data.base_dataset import get_transform
from options import TestOptions


def read_match(match_path):
    match_dict = {}
    with open(match_path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            line_split = line.strip().split(',')
            a_filename = os.path.basename(line_split[0])
            b_filename = os.path.basename(line_split[1])

            if a_filename in match_dict:
                match_dict[a_filename].append(b_filename)
            else:
                match_dict[a_filename] = [b_filename]

    return match_dict


def read_images(image_dir):
    image_dict = {}
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_lower = ext.lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg' or ext_lower == '.bmp' or ext_lower == '.tif':
                image_dict[filename] = os.path.join(path, filename)
    return image_dict


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = models.create_model(opt)

    image_a_dir = os.path.join(opt.dataroot, 'trainB' if opt.direction == 'BtoA' else 'trainA')
    image_b_dir = os.path.join(opt.dataroot, 'trainA' if opt.direction == 'BtoA' else 'trainB')

    # read image
    image_a_dict = read_images(image_a_dir)
    image_b_dict = read_images(image_b_dir)

    # read match
    match_dict = read_match(opt.match)

    # get transform
    transform_A = get_transform(opt, grayscale=False)

    image_a_dict_list = list(image_a_dict.items())

    dry_run = False
    if dry_run:
        import random

        random.Random(123).shuffle(image_a_dict_list)
        image_a_dict_list = image_a_dict_list[:400]

    for A_filename, A_path in tqdm(image_a_dict_list):
        B_filename_list = match_dict[A_filename]
        for B_i, B_filename in enumerate(B_filename_list):
            B_path = image_b_dict[B_filename]

            A_img = Image.open(A_path).convert('RGB')
            A = transform_A(A_img)

            B_img = Image.open(B_path).convert('RGB')
            B = transform_A(B_img)

            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

            A_label = None
            B_label = None

            sp = model(A, A_label, command="encode_sp")
            gl = model(B, B_label, command="encode_gl")
            mix = model(sp, gl, command="decode")
            mix = mix[0]

            mix_img = util.tensor2im(mix, tile=False)

            tag = '' if B_i == 0 else str(B_i)
            fake_dir = os.path.join(opt.result_dir, opt.fake_dir + tag)
            os.makedirs(fake_dir, exist_ok=True)

            # save image
            Image.fromarray(mix_img).save(os.path.join(fake_dir, A_filename))
