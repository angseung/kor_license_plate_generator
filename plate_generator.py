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


def blend_argb_with_rgb(fg: np.ndarray, bg: np.ndarray, row: int, col: int) -> np.ndarray:
    _, mask = cv2.threshold(fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = bg[row: row + h, col: col + w]

    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    added = masked_fg + masked_bg

    return added


class ImageGenerator:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Plate
        self.plate = cv2.imread("plate.jpg")
        self.plate2 = cv2.imread("plate_y.jpg")
        self.plate3 = cv2.imread("plate_g.jpg")
        self.plate_elec = cv2.imread("plate_e.png")
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

        # loading transparent images for electronic car number plate
        file_path = "./num_tr/"
        file_list = os.listdir(file_path)
        self.Number_tr = list()
        self.number_list_tr = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # img[:, :, 3] = 255
            self.Number_tr.append(img)
            self.number_list_tr.append(file[0:-4])

        # loading Char
        file_path = "./char_tr/"
        file_list = os.listdir(file_path)
        self.char_list_tr = list()
        self.Char_tr = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.Char_tr.append(img)
            self.char_list_tr.append(file[0:-4])

    def yellow_long(self, num: int, save: bool = False):
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
            w, h = resion[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = resion[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56

            # character 3
            rand_int = random.randint(0, len(char) - 1)
            label += self.char_list_y[rand_int]
            w, h = char[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = char[rand_int]
            col += 60 + 36

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            w, h = number[rand_int].shape[:2]
            Plate[row : row + w, col : col + h, :] = number[rand_int]
            col += 56
            Plate = random_bright(Plate)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)

            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def electronic_long(self, num: int, save: bool = False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number_tr]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char_tr]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate_elec, (590, 160))
            label = "Z"
            # row -> y , col -> x
            row, col = 28, 80  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56

            # character 3
            rand_int = random.randint(0, len(char) - 1)
            label += self.char_list_tr[rand_int]
            fg = char[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 60 + 36

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_tr[rand_int]
            fg = number[rand_int]
            added = blend_argb_with_rgb(fg, Plate, row, col)
            w, h = added.shape[:2]

            Plate[row : row + w, col : col + h, :] = added
            col += 56
            # Plate = random_bright(Plate)
            Plate = random_bright(Plate)
            # 2자리 번호판 맨뒤에label 전용 X 삽입
            if save:
                # cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)

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
    A.electronic_long(num_img, save=Save)
