import csv
import os
import re

from PIL import Image
import pandas as pd
import os

images = []


def add_image(path_in, path_out):
    images.append((path_in, path_out))


for img in os.listdir('data/lfw/parts_lfw_funneled_gt_images'):
    match = re.match(r'(?:(\w+)+)_(\d{4})\.ppm', img)
    if not match:
        continue
    name = match.group(1)
    print(f'processing {img}')
    
    current_path = f'data/lfw/parts_lfw_funneled_gt_images/{img}'
    bitmap_path = current_path.replace('ppm', 'bmp')
    image = Image.open(current_path)
    image.save(bitmap_path)

    i = match.group(2)
    add_image(
        f'data/lfw/lfw_funneled/{name}/{name}_{i}.jpg',
        bitmap_path)

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow('input output'.split())
    for row in images:
        writer.writerow(row)
