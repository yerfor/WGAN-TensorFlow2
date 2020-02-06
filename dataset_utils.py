import os
import numpy as np
import cv2
import tensorflow as tf


def load_images_to_array(dir_path, img_shape=(64, 64)):
    def _preprocess(mat):
        mat = cv2.resize(mat, img_shape)
        mat = mat / 127.5 - 1
        mat = mat.astype(np.float32)
        return mat

    image_name_lst = os.listdir(dir_path)
    num_image = len(image_name_lst)
    print(num_image)
    img_lst = []
    for idx, image_name in enumerate(image_name_lst[:10000]):
        if idx % 5000 == 0:
            print("图片已经读取{}/{}".format(idx, num_image))
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_name = os.path.join(dir_path, image_name)
            img = cv2.imread(image_name)
            img = _preprocess(img)
            img_lst.append(img)
    img_arr = np.array(img_lst)
    return img_arr


if __name__ == '__main__':
    dir_path = 'database/anime'
    img_arr = load_images_to_array(dir_path)
    ds = tf.data.Dataset.from_tensor_slices(img_arr).batch(32)
    iterator = iter(ds)
    for i in range(10):
        print(next(iterator))
