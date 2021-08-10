import os
import cv2
import numpy as np
import json


def load_validation_images_and_labels(base_path: str, data_type: str):
    # prepare labels
    issame_list = []
    labels_dir = os.path.join(base_path, data_type, "labels")
    label_file_list = os.listdir(labels_dir)
    label_file_list.sort()
    for file_name in label_file_list:
        label_file_path = os.path.join(labels_dir, file_name)
        with open(label_file_path, encoding="utf-8") as f:
            label_json = json.load(f)
            issame_list.append(label_json['is_same'] == 'True')

    # prepare images
    images_dir = os.path.join(base_path, data_type, "images")
    image_file_list = os.listdir(images_dir)
    image_file_list.sort()

    first_image_path = os.path.join(images_dir, image_file_list[0])
    first_image = cv2.imread(first_image_path)
    w, h, c = first_image.shape
    images = np.empty((len(image_file_list), w, h, c))

    for i, file_name in enumerate(image_file_list):
        image_file_path = os.path.join(images_dir, file_name)
        image_org = cv2.imread(image_file_path)
        image_rgb = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)  # 学習はRGBで行われているため
        images[i, ...] = normalize_image(image_rgb)

    return images, issame_list


def normalize_image(image):
    normalizeed_image = image.astype(np.float32)
    normalizeed_image -= 127.5
    normalizeed_image *= 0.0078125

    return normalizeed_image


# 正規化済み（元ロジックでの入力画像）を入力して、グレースケール化した後正規化して返す
def images_to_grayscale(normalized_images):
    result = np.zeros(normalized_images.shape)
    for image_index in range(len(normalized_images)):
        image = denormalize_image(normalized_images[image_index])
        image = to_grayscale(image)
        image = normalize_image(image)
        result[image_index] = image

    return result


def denormalize_image(normalized_image):
    image = normalized_image / 0.0078125
    image += 127.5
    image = image.astype(np.uint8)

    return image

def to_grayscale(rgb_image):
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.concatenate([image, image, image], axis=2)  # 入力の形に合わせて３チャンネルにする
    return image
