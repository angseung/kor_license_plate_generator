from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from plate_generator import ImageGenerator
from class_labels import class_dict_reversed

generator = ImageGenerator(save_path="./DB")

gen_target: np.ndarray = pd.read_excel("gen_plates.xlsx").iloc[:-1, :].to_numpy()
generated_samples = 0

# for normal plate
normal_index = list(range(10, 46 + 1))
normal_index.remove(30)  # 바
normal_index.remove(31)  # 배
normal_index.remove(35)  # 사
normal_index.remove(39)  # 아
normal_index.remove(43)  # 자
gen_normal = gen_target[normal_index, :]
index_mapper = {val: index for index, val in enumerate(generator.char_list)}
index_mapper_tr = {val: index for index, val in enumerate(generator.char_list_tr)}

for i in range(gen_normal.shape[0]):
    gen = gen_normal[i]
    samples_to_gen = gen[4]
    char = class_dict_reversed[int(gen[0])]
    char_w = index_mapper[char]
    char_tr = index_mapper_tr[char]

    if math.isnan(samples_to_gen):
        continue

    for j in range(int(samples_to_gen) // 4):
        print(f"processing {j}th label {gen[1][1]}")
        # generator.white_short_2digits(char_w, False)
        # generator.white_long_2digits(char_w, False)
        # generator.white_long_3digits(char_w, False)
        # generator.electronic_long(char_tr, False)
        generated_samples += 4

# for plates with region, without "배"
region_index = [30, 35, 39, 43]
gen_region = gen_target[50:]
region_mapper_h = {val: index for index, val in enumerate(generator.region_list_y)}
region_mapper_v = {val: index for index, val in enumerate(generator.region_list_py)}
char_mapper_y = {val: index for index, val in enumerate(generator.char_list_y)}

for i in range(gen_region.shape[0]):
    gen = gen_region[i]
    samples_to_gen = gen[4]
    region = class_dict_reversed[int(gen[0])]

    if math.isnan(samples_to_gen):
        continue

    # for short plate
    if len(region) == 1:
        region_w = region_mapper_h[region]

        # generates samples_to_gen samples...
        for j in range(int(samples_to_gen) // 4):
            # generator.yellow_short(region_w, 0, save=True)  # 아
            # generator.yellow_short(region_w, 1, save=True)  # 바
            # generator.yellow_short(region_w, 3, save=True)  # 사
            # generator.yellow_short(region_w, 4, save=True)  # 자
            generated_samples += 4

    elif len(region) == 3:
        region_h = region_mapper_v[region]

        # generates samples_to_gen samples...
        for j in range(int(samples_to_gen) // 4):
            # generator.yellow_long(region_h, 0, save=True)  # 아
            # generator.yellow_long(region_h, 1, save=True)  # 바
            # generator.yellow_long(region_h, 3, save=True)  # 사
            # generator.yellow_long(region_h, 4, save=True)  # 자
            generated_samples += 4