import util
import cv2
import pandas as pd


file_listing = pd.read_csv("train.csv")
with open('train_faceless.txt', 'r') as f:
    facelesses = [l[:-1] for l in f.readlines()]

gen = util.create_gen_from_file_listing(file_listing, facelesses)

for i, l in gen:
    a = l[:, :, 0]
    b = l[:, :, 1]

    cv2.imshow('i', i)
    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey()

