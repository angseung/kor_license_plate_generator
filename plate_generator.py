import os
import random
import argparse
from typing import List, Union, Tuple
import cv2
import numpy as np


def random_bright(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = 0.5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


class ImageGenerator:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Plate
        self.plate = cv2.imread("plate.jpg")
        self.plate2 = cv2.imread("plate_y.jpg")
        self.plate3 = cv2.imread("plate_g.jpg")
        self.new_plate1 = cv2.imread("new_plate1.png")
        self.new_plate2 = cv2.imread("new_plate2.png")
        self.new_plate3 = cv2.imread("new_plate3.png")
        self.new_plate4 = cv2.imread("new_plate4.png")
        self.new_plate8 = cv2.imread("new_plate8.png")

        # loading Number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Number ====================  yellow-two-line  ==========================
        file_path = "./num_y/"
        file_list = os.listdir(file_path)
        self.Number_y = list()
        self.number_list_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_y.append(img)
            self.number_list_y.append(file[0:-4])

        # loading Char
        file_path = "./char1_y/"
        file_list = os.listdir(file_path)
        self.char_list_y = list()
        self.Char1_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_y.append(img)
            self.char_list_y.append(file[0:-4])

        # loading Region
        file_path = "./region_y/"
        file_list = os.listdir(file_path)
        self.Region_y = list()
        self.region_list_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region_y.append(img)
            self.region_list_y.append(file[0:-4])

        # loading vertical Region
        file_path = "./region_py/"
        file_list = os.listdir(file_path)
        self.Region_py = list()
        self.region_list_py = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region_py.append(img)
            self.region_list_py.append(file[0:-4])
        # =========================================================================

        # loading Number ====================  green-two-line  ==========================
        file_path = "./num_g/"
        file_list = os.listdir(file_path)
        self.Number_g = list()
        self.number_list_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_g.append(img)
            self.number_list_g.append(file[0:-4])

        # loading Char
        file_path = "./char1_g/"
        file_list = os.listdir(file_path)
        self.char_list_g = list()
        self.Char1_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_g.append(img)
            self.char_list_g.append(file[0:-4])

        # loading Resion
        file_path = "./region_g/"
        file_list = os.listdir(file_path)
        self.Resion_g = list()
        self.resion_list_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Resion_g.append(img)
            self.resion_list_g.append(file[0:-4])
        # =========================================================================

    def yellow_long(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number_y]
        resion = [cv2.resize(resion, (56, 83)) for resion in self.Region_py]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1_y]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate2, (520 + 56, 110))
            label = "Z"
            # row -> y , col -> x
            row, col = 13, 35  # row + 83, col + 56

            # number 1
            rand_int = random.randint(0, len(resion) - 1)
            label += self.region_list_py[rand_int]
            Plate[row: row + 83, col: col + 56, :] = resion[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56

            # character 3
            rand_int = random.randint(0, len(char) - 1)
            label += self.char_list_y[rand_int]
            Plate[row: row + 83, col: col + 60, :] = char[rand_int]
            col += 60 + 36

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row: row + 83, col: col + 56, :] = number[rand_int]
            col += 56
            Plate = random_bright(Plate)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)

            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", help="save image directory", type=str, default="./DB/"
    )
    parser.add_argument("-n", "--num", help="number of image", type=int)
    parser.add_argument("-s", "--save", help="save or imshow", type=bool, default=True)
    args = parser.parse_args()

    img_dir = args.img_dir
    A = ImageGenerator(img_dir)

    num_img = args.num
    Save = args.save

    A.yellow_long(num_img, save=Save)
