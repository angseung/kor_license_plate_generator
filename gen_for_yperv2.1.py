import os
import platform
import math
import numpy as np
import pandas as pd
from plate_generator import ImageGenerator
from class_labels import class_dict_reversed

if "Windows" in platform.platform():
    save_path = "./addons_v1.2"
elif "Linux" in platform.platform():
    save_path = "/data_yper/addons_v1.2"
else:
    save_path = "./addons_v1.2"

generator = ImageGenerator(
    save_path=save_path,
    resize_opt=True,
    resize_scale=(1.0, 2.0),
    bright=True,
    perspective=True,
    mode="auto",
    rotate=True,
    angle="auto",
    remove_bg=False,
    debug=False,
)

if not os.path.isdir(f"{save_path}"):
    os.makedirs(f"{save_path}/images/train", exist_ok=True)
    os.makedirs(f"{save_path}/labels/train", exist_ok=True)

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
char_mapper = {val: index for index, val in enumerate(generator.char_list)}
char_mapper_tr = {val: index for index, val in enumerate(generator.char_list_tr)}

for i in range(gen_normal.shape[0]):
    gen = gen_normal[i]
    samples_to_gen = gen[4]
    char = class_dict_reversed[int(gen[0])]
    char_w = char_mapper[char]
    char_tr = char_mapper_tr[char]

    if math.isnan(samples_to_gen) or samples_to_gen <= 0:
        continue

    print(f"processing label {gen[1][1]}")
    for j in range(int(samples_to_gen) // 4):
        generator.white_short_2digits(char_w, True)
        generator.white_long_2digits(char_w, True)
        generator.white_long_3digits(char_w, True)
        generator.electronic_long(char_tr, True)
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

    if math.isnan(samples_to_gen) or samples_to_gen <= 0:
        continue

    # for short plate
    if len(region) == 1:
        region_w = region_mapper_h[region]

        # generates samples_to_gen samples...
        for j in range(int(samples_to_gen) // 5):
            generator.yellow_short(region_w, 0, save=True)  # 아
            generator.yellow_short(region_w, 1, save=True)  # 바
            generator.yellow_short(region_w, 3, save=True)  # 사
            generator.yellow_short(region_w, 4, save=True)  # 자
            generated_samples += 4

    elif len(region) == 3:
        region_h = region_mapper_v[region]

        # generates samples_to_gen samples...
        for j in range(int(samples_to_gen) // 5):
            generator.yellow_long(region_h, 0, save=True)  # 아
            generator.yellow_long(region_h, 1, save=True)  # 바
            generator.yellow_long(region_h, 3, save=True)  # 사
            generator.yellow_long(region_h, 4, save=True)  # 자
            generated_samples += 4

# save remainder
samples_to_gen_parcel = int(samples_to_gen) // 5

# for rent car plates
gen_rents = gen_target[[47, 48, 49], :]

for i in range(gen_rents.shape[0]):
    gen = gen_rents[i]
    samples_to_gen = gen[4]
    char = class_dict_reversed[int(gen[0])]
    char_w = char_mapper[char]
    char_tr = char_mapper_tr[char]

    if math.isnan(samples_to_gen) or samples_to_gen <= 0:
        continue

    print(f"processing label {gen[1][1]}")

    for j in range(int(samples_to_gen) // 4):
        generator.white_short_2digits(char_w, True)
        generator.white_long_2digits(char_w, True)
        generator.white_long_3digits(char_w, True)
        generator.electronic_long(char_tr, True)
        generated_samples += 4

# for parcel car plates
gen_parcel = gen_target[31, :]
char = class_dict_reversed[int(gen_parcel[0])]

char_y = char_mapper_y[char]
char_tr = char_mapper_tr[char]

for i in range(samples_to_gen_parcel):
    # for all regions of yellow plates
    for j in range(17):
        generator.yellow_long(j, char_y, save=True)
        generator.yellow_short(j, char_y, save=True)
        generated_samples += 2

# green short plates
for region in range(16):
    print(f"gen green short {region}/16")
    for char in range(37):
        for _ in range(10):
            generator.green_short(region, char, save=True)
