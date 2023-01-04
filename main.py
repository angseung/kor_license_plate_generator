import numpy as np
from plate_generator import ImageGenerator

generator = ImageGenerator(save_path="./DB")
iters = 100

# 2digit white short
for char in range(39):
    for _ in range(iters):
        generator.white_short_2digits(char, True)

# 2digit white long
for char in range(39):
    for _ in range(iters):
        generator.white_long_2digits(char, True)

# 3digit white long
for char in range(39):
    for _ in range(iters):
        generator.white_long_3digits(char, True)

# electronic long
for char in range(39):
    for _ in range(iters):
        generator.electronic_long(char, True)

# yellow long
for char in range(40):
    for region in range(17):
        for _ in range(iters):
            generator.yellow_long(region, char, True)

# yellow short
for char in range(40):
    for region in range(17):
        for _ in range(iters):
            generator.yellow_short(region, char, True)

# green short
for char in range(39):
    for region in range(16):
        for _ in range(iters):
            generator.green_short(region, char, True)
