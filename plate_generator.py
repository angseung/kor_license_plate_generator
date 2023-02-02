import os
import random
from datetime import datetime
import argparse
from typing import List, Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
from class_labels import class_dict


def remove_white_bg(img: np.ndarray) -> np.ndarray:
    # Convert image to image gray
    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, 250, 255, cv2.THRESH_BINARY_INV)

    # Using cv2.split() to split channels
    # of coloured image
    b, g, r = cv2.split(img)

    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]

    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    return cv2.merge(rgba, 4)


def warp_point(x: int, y: int, M: np.ndarray) -> Tuple[int, int]:
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    )


def random_perspective(
    img: np.ndarray,
    labels: np.ndarray,
    mode: str = "auto",
    max_pad_order: Tuple[int, int] = (4, 8),
) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(datetime.now().timestamp())

    H, W = img.shape[:2]
    max_pad_h = H // max_pad_order[0]
    max_pad_w = W // max_pad_order[1]

    if mode == "auto":
        mode_list = ["top", "bottom", "left", "right"]
        selected = random.randint(0, 3)
        mode = mode_list[selected]

    labels = label_yolo2voc(labels, H, W)

    if mode in ["top", "bottom"]:
        pad_l, pad_r = (random.randint(1, max_pad_w), random.randint(1, max_pad_w))

        img_padded = np.zeros([H, W + pad_l + pad_r, 3], dtype=np.uint8)
        img_padded[:, :, :] = 255
        img_padded[:, pad_l:-pad_r, :] = img

        if mode == "top":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[pad_l, 0], [W + pad_l, 0], [2 * pad_l, H], [W + pad_l - pad_r, H]]
            )

        elif mode == "bottom":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[2 * pad_l, 0], [W + pad_l - pad_r, 0], [pad_l, H], [W + pad_l, H]]
            )

    elif mode in ["left", "right"]:
        pad_top, pad_bottom = (
            random.randint(1, max_pad_h),
            random.randint(1, max_pad_h),
        )
        img_padded = np.zeros([H + pad_top + pad_bottom, W, 3], dtype=np.uint8)
        img_padded[:, :, :] = 255
        img_padded[pad_top:-pad_bottom, :, :] = img

        if mode == "left":
            point_before = np.float32(
                [
                    [0, 2 * pad_top],
                    [W, pad_top],
                    [0, H + pad_top - pad_bottom],
                    [W, H + pad_top],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

        elif mode == "right":
            point_before = np.float32(
                [
                    [0, pad_top],
                    [W, 2 * pad_top],
                    [0, H + pad_top],
                    [W, H + pad_top - pad_bottom],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

    mtrx = cv2.getPerspectiveTransform(point_before, point_after)
    result = cv2.warpPerspective(img_padded, mtrx, img_padded.shape[:2][::-1])

    for i, label in enumerate(labels):
        xtl, ytl, xbr, ybr = label.tolist()[1:]

        if mode in ["top", "bottom"]:
            xtl += pad_l
            xbr += pad_l

        elif mode in ["left", "right"]:
            ytl += pad_top
            ybr += pad_top

        xtl_new, ytl_new = warp_point(xtl, ytl, mtrx)
        xtr_new, ytr_new = warp_point(xbr, ytl, mtrx)
        xbl_new, ybl_new = warp_point(xtl, ybr, mtrx)
        xbr_new, ybr_new = warp_point(xbr, ybr, mtrx)

        labels[i, 1:] = np.uint32(
            [
                (xtl_new + xbl_new) // 2,
                (ytl_new + ytr_new) // 2,
                (xtr_new + xbr_new) // 2,
                (ybl_new + ybr_new) // 2,
            ]
        )

    labels = label_voc2yolo(labels, *result.shape[:2])

    return result, labels


def parse_label(fname: str) -> np.ndarray:
    """
    parses the label file, then converts it to np.ndarray type
    Args:
        fname: label file name

    Returns: label as np.ndarray

    """
    with open(fname, encoding="utf-8") as f:
        bboxes = f.readlines()
        label = []

    for bbox in bboxes:
        label.append(bbox.split())

    return np.array(label, dtype=np.float64)


def random_bright(img: np.ndarray) -> np.ndarray:
    random.seed(datetime.now().timestamp())

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


def blend_argb_with_argb(
    fg: np.ndarray, bg: np.ndarray, row: int, col: int
) -> np.ndarray:

    assert fg.shape[2] == 4 and bg.shape[2] == 4

    h, w = fg.shape[:2]

    cropped_bg = bg[row : row + h, col : col + w, :]  # BGRA
    bA = cropped_bg[:, :, 0]
    gA = cropped_bg[:, :, 1]
    rA = cropped_bg[:, :, 2]
    aA = cropped_bg[:, :, 3]

    bB = fg[:, :, 0]
    gB = fg[:, :, 1]
    rB = fg[:, :, 2]
    aB = fg[:, :, 3]

    rOut = (rA * aA / 255) + (rB * aB * (255 - aA) / (255 * 255))
    gOut = (gA * aA / 255) + (gB * aB * (255 - aA) / (255 * 255))
    bOut = (bA * aA / 255) + (bB * aB * (255 - aA) / (255 * 255))
    aOut = aA + (aB * (255 - aA) / 255)

    bg[row : row + h, col : col + w, 0] = bOut
    bg[row : row + h, col : col + w, 1] = gOut
    bg[row : row + h, col : col + w, 2] = rOut
    bg[row : row + h, col : col + w, 3] = aOut

    return bg


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


def convert_bbox_to_label(bboxes: List[str]) -> np.ndarray:
    labels = np.zeros((len(bboxes), 5))

    for i, bbox in enumerate(bboxes):
        labels[i, :] = np.array(bbox.split())

    return labels


def write_label_from_str(target_dir: str, fname: str, *bboxes: List[str]) -> None:
    num_boxes = len(bboxes)

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            f.write(f"{bboxes[i]}\n")


def write_label(target_dir: str, fname: str, bboxes: np.ndarray) -> None:
    """
    exports np.ndarray label to txt file
    Args:
        target_dir: save dir for label file
        fname: file name of label
        bboxes: annotation information, np.ndarray type

    Returns: None

    """
    num_boxes = bboxes.shape[0]

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            target_str = f"{int(bboxes[i][0])} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]} {bboxes[i][4]}"
            f.write(f"{target_str}\n")


def random_resize(
    img: np.ndarray,
    label: Optional[Union[np.ndarray, None]] = None,
    scale_min: Union[int, float] = 0.75,
    scale_max: Union[int, float] = 2.5,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    random.seed(datetime.now().timestamp())
    scaled = random.uniform(scale_min, scale_max)
    h, w = img.shape[:2]

    if h > w:
        ratio = h / w
        w_scaled = w * scaled
        h_scaled = w_scaled * ratio

    else:
        ratio = w / h
        h_scaled = h * scaled
        w_scaled = h_scaled * ratio

    size = int(w_scaled), int(h_scaled)

    if label is not None:
        label = label_yolo2voc(label, h, w).astype(np.float64)
        label[:, 1:] *= scaled
        label = label_voc2yolo(label, h_scaled, w_scaled)

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA), label

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def label_yolo2voc(label_yolo: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from yolo to voc
    Args:
        label_yolo: (x_center, y_center, w, h), normalized
        h: img height
        w: img width

    Returns: (xtl, ytl, xbr, ybr)

    """
    label_voc = np.zeros(label_yolo.shape, dtype=np.float64)
    label_voc[:, 0] = label_yolo[:, 0]

    label_yolo_temp = label_yolo.copy()
    label_yolo_temp[:, [1, 3]] *= w
    label_yolo_temp[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_voc[:, 1] = label_yolo_temp[:, 1] - 0.5 * label_yolo_temp[:, 3]
    label_voc[:, 2] = label_yolo_temp[:, 2] - 0.5 * label_yolo_temp[:, 4]

    # convert width, height to xbr, ybr
    label_voc[:, 3] = label_voc[:, 1] + label_yolo_temp[:, 3]
    label_voc[:, 4] = label_voc[:, 2] + label_yolo_temp[:, 4]

    return label_voc.astype(np.uint32)


def label_voc2yolo(label_voc: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from voc to yolo
    Args:
        label_voc: (xtl, ytl, xbr, ybr)
        h: img heights
        w: img width

    Returns: (x_center, y_center, w, h), normalized

    """
    label_yolo = np.zeros(label_voc.shape, dtype=np.float64)
    label_yolo[:, 0] = label_voc[:, 0]

    # convert xtl, ytl to x_center, y_center
    label_yolo[:, 1] = 0.5 * (label_voc[:, 1] + label_voc[:, 3])
    label_yolo[:, 2] = 0.5 * (label_voc[:, 2] + label_voc[:, 4])

    # convert xbr, ybr to width, height
    label_yolo[:, 3] = label_voc[:, 3] - label_voc[:, 1]
    label_yolo[:, 4] = label_voc[:, 4] - label_voc[:, 2]

    # normalize
    label_yolo[:, [1, 3]] /= w
    label_yolo[:, [2, 4]] /= h

    return label_yolo


def draw_bbox_on_img(
    img: np.ndarray, label: np.ndarray, is_voc: bool = False
) -> np.ndarray:
    if not is_voc:
        label = label_yolo2voc(label, *(img.shape[:2]))

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(label.shape[0]):
        pos = tuple(label[i][1:].tolist())
        draw.rectangle(pos, outline=(0, 0, 0), width=3)

    return np.asarray(img)


def save_img_label(
    img: np.ndarray,
    labels: np.ndarray,
    target_dir: str,
    fname: str,
    resize: bool = True,
    resize_scale: Tuple[float, float] = (1.0, 3.0),
    bright: bool = True,
    perspective: bool = True,
    mode: str = "auto",
    remove_bg: bool = False,
    debug: bool = False,
):
    if resize:
        img, labels = random_resize(
            img, labels, scale_min=resize_scale[0], scale_max=resize_scale[1]
        )

    if perspective:
        img, labels = random_perspective(img, labels, mode=mode)

    if remove_bg:
        img = remove_white_bg(img)

    if bright:
        img = random_bright(img)

    if debug:
        img = draw_bbox_on_img(img=img, label=labels)

    cv2.imwrite(target_dir + "/images/train/" + fname + ".png", img)
    write_label(target_dir + "/labels/train", fname, labels)


class ImageGenerator:
    def __init__(
        self,
        save_path: str,
        resize_opt: Optional[bool] = True,
        resize_scale: Optional[Tuple[float, float]] = (1.0, 3.0),
        bright: Optional[bool] = True,
        perspective: Optional[bool] = True,
        mode: Optional[str] = "auto",
        remove_bg: Optional[bool] = False,
        debug: Optional[bool] = False,
    ):
        self.random_resize = resize_opt
        self.resize_scale = resize_scale
        self.bright = bright
        self.perspective = perspective
        self.mode = mode
        self.remove_bg = remove_bg
        self.debug = debug
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
        file_list = sorted(os.listdir(file_path))
        self.number = list()
        self.number_list = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = sorted(os.listdir(file_path))
        self.char_list = list()
        self.char1 = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Number ====================  yellow-two-line  ==========================
        file_path = "./num_y/"
        file_list = sorted(os.listdir(file_path))
        self.number_y = list()
        self.number_list_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number_y.append(img)
            self.number_list_y.append(file[0:-4])

        # loading Char
        file_path = "./char1_y/"
        file_list = sorted(os.listdir(file_path))
        self.char_list_y = list()
        self.char1_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1_y.append(img)
            self.char_list_y.append(file[0:-4])

        # loading Region
        file_path = "./region_y/"
        file_list = sorted(os.listdir(file_path))
        self.region_y = list()
        self.region_list_y = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.region_y.append(img)
            self.region_list_y.append(file[0:-4])

        # loading vertical Region
        file_path = "./region_py/"
        file_list = sorted(os.listdir(file_path))
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
        file_list = sorted(os.listdir(file_path))
        self.number_g = list()
        self.number_list_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.number_g.append(img)
            self.number_list_g.append(file[0:-4])

        # loading Char
        file_path = "./char1_g/"
        file_list = sorted(os.listdir(file_path))
        self.char_list_g = list()
        self.char1_g = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.char1_g.append(img)
            self.char_list_g.append(file[0:-4])

        # loading green Region
        file_path = "./region_g/"
        file_list = sorted(os.listdir(file_path))
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
        file_list = sorted(os.listdir(file_path))
        self.number_tr = list()
        self.number_list_tr = list()

        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.number_tr.append(img)
            self.number_list_tr.append(file[0:-4])

        # loading Char
        file_path = "./char_tr/"
        file_list = sorted(os.listdir(file_path))
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
        label = "GSR"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def yellow_long(self, region_label: int, char_label: int, save: bool = True):
        number_y = [cv2.resize(number, (56, 83)) for number in self.number_y]
        region_py = [cv2.resize(region, (56, 83)) for region in self.region_py]
        char_y = [cv2.resize(char1, (60, 83)) for char1 in self.char1_y]

        plate = cv2.resize(self.plate2, (520 + 56, 110))
        label = "YLR"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def yellow_short(self, region_label: int, char_label: int, save: bool = True):
        number_y = [cv2.resize(number, (44, 60)) for number in self.number_y]
        number2_y = [cv2.resize(number, (64, 90)) for number in self.number_y]
        region_y = [cv2.resize(region, (88, 60)) for region in self.region_y]
        char_y = [cv2.resize(char1, (64, 62)) for char1 in self.char1_y]

        plate = cv2.resize(self.plate2, (336, 170))
        label = "YSR"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def electronic_long(self, char_label: int, save: bool = True):
        number_tr = [cv2.resize(number, (56, 83)) for number in self.number_tr]
        char_tr = [cv2.resize(char1, (60, 83)) for char1 in self.char_tr]
        bboxes = []

        plate = cv2.resize(self.plate_elec, (590, 160))
        label = "EL"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def white_long_2digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (56, 83)) for number in self.number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.char1]

        plate = cv2.resize(self.new_plate1, (520, 110))
        bboxes = []
        label = "WL2"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def white_long_3digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (56, 83)) for number in self.number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.char1]

        plate = cv2.resize(self.plate, (520 + 56, 110))
        bboxes = []
        label = "WL3"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )

    def white_short_2digits(self, char_label: int, save: bool = True):
        number = [cv2.resize(number, (45, 83)) for number in self.number]
        char = [cv2.resize(char1, (49, 70)) for char1 in self.char1]

        plate = cv2.resize(self.plate, (355, 155))
        bboxes = []
        label = "WS2"
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

        labels = convert_bbox_to_label(bboxes)

        save_img_label(
            img=plate,
            labels=labels,
            target_dir=self.save_path,
            fname=label,
            resize=self.random_resize,
            resize_scale=self.resize_scale,
            bright=self.bright,
            perspective=self.perspective,
            mode=self.mode,
            remove_bg=self.remove_bg,
            debug=self.debug,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", help="save image directory", type=str, default="./DB_new"
    )
    parser.add_argument("-n", "--num", help="number of image", type=int)
    parser.add_argument("-s", "--save", help="save or imshow", type=bool, default=True)
    args = parser.parse_args()

    img_dir = args.img_dir

    A = ImageGenerator(
        save_path=img_dir,
        resize_opt=True,
        resize_scale=(1.0, 3.0),
        bright=True,
        perspective=True,
        mode="auto",
        remove_bg=False,
        debug=False,
    )

    if not os.path.isdir(f"{img_dir}"):
        os.makedirs(f"{img_dir}/images/train", exist_ok=True)
        os.makedirs(f"{img_dir}/labels/train", exist_ok=True)

    num_img = args.num
    Save = args.save

    A.yellow_long(0, 0, save=Save)
    A.electronic_long(0, save=Save)
    A.white_long_2digits(0, save=Save)
    A.white_long_3digits(0, save=Save)
    A.white_short_2digits(0, save=Save)
    A.yellow_short(0, 0, save=Save)
    A.green_short(0, 0, save=Save)
