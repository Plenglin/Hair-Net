import csv
import os
import re

from PIL import Image
import os
import random


TEST_PCT = 0.1

images = []
faceless = []

def add_image(path_in, path_out):
    images.append((path_in, path_out))


for img in os.listdir("data/lfw/parts_lfw_funneled_gt_images_bmp"):
    match = re.match(r"(?:(\w+)+)_(\d{4})\.bmp", img)
    if not match:
        continue
    name = match.group(1)
    print(f"processing {img}")

    #current_path = f"data/lfw/parts_lfw_funneled_gt_images/{img}"
    #bitmap_path = current_path.replace("ppm", "bmp")
    #image = Image.open(current_path)
    #image.save(bitmap_path)

    i = match.group(2)
    add_image(f"data/lfw/lfw_funneled/{name}/{name}_{i}.jpg", "data/lfw/parts_lfw_funneled_gt_images_bmp/" + img)

for img in os.listdir('data/faceless'):
    faceless.append('data/faceless/' + img)

random.shuffle(images)
random.shuffle(faceless)
split_index_imgs = int(len(images) * TEST_PCT)
split_index_faceless = int(len(faceless) * TEST_PCT)
test = images[:split_index_imgs]
train = images[split_index_imgs:]

test_faceless = faceless[:split_index_faceless]
train_faceless = faceless[split_index_faceless:]

with open("train.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow("input output".split())
    for row in train:
        writer.writerow(row)

with open("test.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow("input output".split())
    for row in test:
        writer.writerow(row)

with open("train_faceless.txt", "w", newline="\n") as file:
    for row in train_faceless:
        file.write(row + "\n")

with open("test_faceless.txt", "w", newline="\n") as file:
    for row in test_faceless:
        file.write(row + "\n")
