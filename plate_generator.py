import os
import random
import argparse
from typing import List, Union, Tuple
import cv2
import numpy as np
from class_labels import class_dict


def random_bright(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = 0.5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img


def blend_argb_with_rgb(
    fg: np.ndarray, bg: np.ndarray, row: int, col: int
) -> np.ndarray:
    _, mask = cv2.threshold(fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = bg[row : row + h, col : col + w]

    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    blended = masked_fg + masked_bg

    return blended


def make_bboxes(
    img: np.ndarray, obj: np.ndarray, label: int, ytl: int, xtl: int
) -> str:
    h, w = obj.shape[:2]
    xbr = xtl + w
    ybr = ytl + h
    center_x = (xtl + xbr) / 2.0
    center_y = (ytl + ybr) / 2.0

    h_bg, w_bg = img.shape[:2]

    # yolo format (x_center, y_center, width, height)
    return f"{label} {center_x / w_bg} {center_y / h_bg} {w / w_bg} {h / h_bg}"


def write_label(target_dir: str, fname: str, *bboxes: List[str]) -> None:
    num_boxes = len(bboxes)

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            f.write(f"{bboxes[i]}\n")


class ImageGenerator:
    def __init__(self, save_path: str):
        self.save_path = save_path

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
        self.class_dict = class_dict

        # loading Number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.number = list()
        self.number_list = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.char1 = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Number ====================  yellow-two-line  ==========================
        file_path = "./num_y/"
        file_list = os.listdir(file_path)
        self.number_y = list()
        self.number_list_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number_y.append(img)
            self.number_list_y.append(file[0:-4])

        # loading Char
        file_path = "./char1_y/"
        file_list = os.listdir(file_path)
        self.char_list_y = list()
        self.char1_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1_y.append(img)
            self.char_list_y.append(file[0:-4])

        # loading Region
        file_path = "./region_y/"
        file_list = os.listdir(file_path)
        self.region_y = list()
        self.region_list_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.region_y.append(img)
            self.region_list_y.append(file[0:-4])

        # loading vertical Region
        file_path = "./region_py/"
        file_list = os.listdir(file_path)
        self.region_py = list()
        self.region_list_py = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.region_py.append(img)
            self.region_list_py.append(file[0:-4])
        # =========================================================================

        # loading Number ====================  green-two-line  ==========================
        file_path = "./num_g/"
        file_list = os.listdir(file_path)
        self.number_g = list()
        self.number_list_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number_g.append(img)
            self.number_list_g.append(file[0:-4])

        # loading Char
        file_path = "./char1_g/"
        file_list = os.listdir(file_path)
        self.char_list_g = list()
        self.char1_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1_g.append(img)
            self.char_list_g.append(file[0:-4])

        # loading green Region
        file_path = "./region_g/"
        file_list = os.listdir(file_path)
        self.region_g = list()
        self.region_list_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.region_g.append(img)
            self.region_list_g.append(file[0:-4])
        # =========================================================================

        # loading transparent images for electronic car number plate
        file_path = "./num_tr/"
        file_list = os.listdir(file_path)
        self.number_tr = list()
        self.number_list_tr = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.number_tr.append(img)
            self.number_list_tr.append(file[0:-4])

        # loading Char
        file_path = "./char_tr/"
        file_list = os.listdir(file_path)
        self.char_list_tr = list()
        self.char_tr = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.char_tr.append(img)
            self.char_list_tr.append(file[0:-4])

    def green_short(self, region_label: int, char_label: int, save: bool = True):
        number_g = [cv2.resize(number, (44, 60)) for number in self.number_g]
        number2_g = [cv2.resize(number, (64, 90)) for number in self.number_g]
        resion_g = [cv2.resize(resion, (88, 60)) for resion in self.region_g]
        char_g = [cv2.resize(char1, (64, 62)) for char1 in self.char1_g]

        plate = cv2.resize(self.plate3, (336, 170))
        label = "Z"
        # row -> y , col -> x
        row, col = 8, 76  # row + 83, col + 56
        bboxes = []

        # region
        label += self.region_list_g[region_label]
        w, h = resion_g[region_label].shape[:2]
        plate[row : row + w, col : col + h, :] = resion_g[region_label]
        bboxes.append(
            make_bboxes(
                plate,
                resion_g[region_label],
                self.class_dict[self.region_list_g[region_label]],
                row,
                col,
            )
        )
        col += 88 + 4

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_g[rand_int]
        bboxes.append(make_bboxes(plate, number_g[rand_int], rand_int, row, col))
        col += 44

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_g[rand_int]
        bboxes.append(make_bboxes(plate, number_g[rand_int], rand_int, row, col))

        row, col = 72, 8

        # character 3
        label += self.char_list_g[char_label]
        w, h = char_g[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char_g[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char_g[char_label],
                self.class_dict[self.char_list_g[char_label]],
                row,
                col,
            )
        )
        col += 64

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number2_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_g[rand_int]
        bboxes.append(make_bboxes(plate, number2_g[rand_int], rand_int, row, col))
        col += 64

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number2_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_g[rand_int]
        bboxes.append(make_bboxes(plate, number2_g[rand_int], rand_int, row, col))
        col += 64

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number2_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_g[rand_int]
        bboxes.append(make_bboxes(plate, number2_g[rand_int], rand_int, row, col))
        col += 64

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list_g[rand_int]
        w, h = number2_g[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_g[rand_int]
        bboxes.append(make_bboxes(plate, number2_g[rand_int], rand_int, row, col))
        col += 64
        plate = random_bright(plate)

        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + ".jpg", plate)
            write_label(self.save_path + "/labels/train", label, *bboxes)

        else:
            pass

    def yellow_long(self, region_label: int, char_label: int, save: bool = True):
        number_y = [cv2.resize(number, (56, 83)) for number in self.number_y]
        region_py = [cv2.resize(region, (56, 83)) for region in self.region_py]
        char_y = [cv2.resize(char1, (60, 83)) for char1 in self.char1_y]

        plate = cv2.resize(self.plate2, (520 + 56, 110))
        label = "Z"
        # row -> y , col -> x
        row, col = 13, 35  # row + 83, col + 56
        bboxes = []

        # number 1
        label += self.region_list_py[region_label]
        w, h = region_py[region_label].shape[:2]
        plate[row : row + w, col : col + h, :] = region_py[region_label]
        bboxes.append(
            make_bboxes(
                plate,
                region_py[region_label],
                self.class_dict[self.region_list_py[region_label]],
                row,
                col,
            )
        )
        col += 56

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56

        # number 3
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56

        # character 3
        label += self.char_list_y[char_label]
        w, h = char_y[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char_y[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char_y[char_label],
                self.class_dict[self.char_list_y[char_label]],
                row,
                col,
            )
        )
        col += 60 + 36

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 56
        plate = random_bright(plate)

        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + ".jpg", plate)
            write_label(self.save_path + "/labels/train", label, *bboxes)

        else:
            pass

    def yellow_short(self, region_label: int, char_label: int, save: bool = True):
        number_y = [cv2.resize(number, (44, 60)) for number in self.number_y]
        number2_y = [cv2.resize(number, (64, 90)) for number in self.number_y]
        region_y = [cv2.resize(region, (88, 60)) for region in self.region_y]
        char_y = [cv2.resize(char1, (64, 62)) for char1 in self.char1_y]

        plate = cv2.resize(self.plate2, (336, 170))
        label = "Z"
        # row -> y , col -> x
        row, col = 8, 76  # row + 83, col + 56
        bboxes = []

        # region
        label += self.region_list_y[region_label]
        w, h = region_y[region_label].shape[:2]
        plate[row : row + w, col : col + h, :] = region_y[region_label]
        bboxes.append(
            make_bboxes(
                plate,
                region_y[region_label],
                self.class_dict[self.region_list_y[region_label]],
                row,
                col,
            )
        )
        col += 88 + 4

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))
        col += 44

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number_y[rand_int]
        bboxes.append(make_bboxes(plate, number_y[rand_int], rand_int, row, col))

        row, col = 72, 8

        # character 3
        label += self.char_list_y[char_label]
        w, h = char_y[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char_y[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char_y[char_label],
                self.class_dict[self.char_list_y[char_label]],
                row,
                col,
            )
        )
        col += 64

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number2_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_y[rand_int]
        bboxes.append(make_bboxes(plate, number2_y[rand_int], rand_int, row, col))
        col += 64

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number2_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_y[rand_int]
        bboxes.append(make_bboxes(plate, number2_y[rand_int], rand_int, row, col))
        col += 64

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number2_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_y[rand_int]
        bboxes.append(make_bboxes(plate, number2_y[rand_int], rand_int, row, col))
        col += 64

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list_y[rand_int]
        w, h = number2_y[rand_int].shape[:2]
        plate[row : row + w, col : col + h, :] = number2_y[rand_int]
        bboxes.append(make_bboxes(plate, number2_y[rand_int], rand_int, row, col))
        col += 64
        plate = random_bright(plate)

        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + ".jpg", plate)
            write_label(self.save_path + "/labels/train", label, *bboxes)

        else:
            pass

    def electronic_long(self, char_label: int, save: bool = True):
        number_tr = [cv2.resize(number, (56, 83)) for number in self.number_tr]
        char_tr = [cv2.resize(char1, (60, 83)) for char1 in self.char_tr]
        bboxes = []

        plate = cv2.resize(self.plate_elec, (590, 160))
        label = "Z"
        # row -> y , col -> x
        row, col = 28, 80  # row + 83, col + 56

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        # character 3
        label += self.char_list_tr[char_label]
        fg = char_tr[char_label]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(
            make_bboxes(
                plate,
                char_tr[char_label],
                self.class_dict[self.char_list_tr[char_label]],
                row,
                col,
            )
        )
        col += 60 + 36

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list_tr[rand_int]
        fg = number_tr[rand_int]
        added = blend_argb_with_rgb(fg, plate, row, col)
        w, h = added.shape[:2]

        plate[row : row + w, col : col + h, :] = added
        bboxes.append(make_bboxes(plate, number_tr[rand_int], rand_int, row, col))
        col += 56

        plate = random_bright(plate)

        # 2자리 번호판 맨 뒤에 label 전용 X 삽입
        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + "X.jpg", plate)
            write_label(self.save_path + "/labels/train", f"{label}X", *bboxes)

        else:
            pass

    def white_long_2digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (56, 83)) for number in self.number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.char1]

        plate = cv2.resize(self.new_plate1, (520, 110))
        bboxes = []
        label = "Z"
        # row -> y , col -> x
        row, col = 13, 35  # row + 83, col + 56

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # character 3
        label += self.char_list[char_label]
        w, h = char[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char[char_label],
                self.class_dict[self.char_list[char_label]],
                row,
                col,
            )
        )
        col += 60 + 36

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        plate = random_bright(plate)

        # 2자리 번호판 맨 뒤에 label 전용 X 삽입
        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + "X.jpg", plate)
            write_label(self.save_path + "/labels/train", f"{label}X", *bboxes)

        else:
            pass

    def white_long_3digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (56, 83)) for number in self.number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.char1]

        plate = cv2.resize(self.plate, (520 + 56, 110))
        bboxes = []
        label = "Z"
        # row -> y , col -> x
        row, col = 13, 35  # row + 83, col + 56

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 3
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # character 3
        label += self.char_list[char_label]
        w, h = char[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char[char_label],
                self.class_dict[self.char_list[char_label]],
                row,
                col,
            )
        )
        col += 60 + 36

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 56, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 56

        plate = random_bright(plate)

        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + ".jpg", plate)
            write_label(self.save_path + "/labels/train", label, *bboxes)

        else:
            pass

    def white_short_2digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (45, 83)) for number in self.number]
        char = [cv2.resize(char1, (49, 70)) for char1 in self.char1]

        plate = cv2.resize(self.plate, (355, 155))
        bboxes = []
        label = "Z"
        # row -> y , col -> x
        row, col = 46, 10  # row + 83, col + 56

        # number 1
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45

        # number 2
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45

        # character 3
        label += self.char_list[char_label]
        w, h = char[char_label].shape[:2]
        plate[row : row + w, col : col + h, :] = char[char_label]
        bboxes.append(
            make_bboxes(
                plate,
                char[char_label],
                self.class_dict[self.char_list[char_label]],
                row,
                col,
            )
        )
        col += 49 + 2

        # number 4
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45 + 2

        # number 5
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45

        # number 6
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45

        # number 7
        rand_int = random.randint(0, 9)
        label += self.number_list[rand_int]
        plate[row : row + 83, col : col + 45, :] = number[rand_int]
        bboxes.append(make_bboxes(plate, number[rand_int], rand_int, row, col))
        col += 45

        plate = random_bright(plate)

        # 2자리 번호판 맨뒤에label 전용 X 삽입
        if save:
            cv2.imwrite(self.save_path + "/images/train/" + label + "X.jpg", plate)
            write_label(self.save_path + "/labels/train", f"{label}X", *bboxes)

        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", help="save image directory", type=str, default="./DB_new"
    )
    parser.add_argument("-n", "--num", help="number of image", type=int)
    parser.add_argument("-s", "--save", help="save or imshow", type=bool, default=True)
    args = parser.parse_args()

    img_dir = args.img_dir
    A = ImageGenerator(img_dir)

    num_img = args.num
    Save = args.save

    A.yellow_long(0, 0, save=Save)
    A.electronic_long(0, save=Save)
    A.white_long_2digits(0, save=Save)
    A.white_long_3digits(0, save=Save)
    A.white_short_2digits(0, save=Save)
    A.yellow_short(0, 0, save=Save)
    A.green_short(0, 0, save=Save)
