from plate_generator import ImageGenerator

generator = ImageGenerator(save_path="./DB_new")

# # 2digit white short
# for char in range(35):
#     for _ in range(320):
#         generator.white_short_2digits(char, True)
#
# # 2digit white long
# for char in range(35):
#     for _ in range(320):
#         generator.white_long_2digits(char, True)
#
# # 3digit white long
# for char in range(35):
#     for _ in range(320):
#         generator.white_long_3digits(char, True)

# electronic long
for char in range(40):
    for _ in range(320):
        generator.electronic_long(char, True)

# yellow long
for char in range(5):
    for region in range(17):
        for _ in range(200):
            generator.yellow_long(region, char, True)

# yellow short
for char in range(5):
    for region in range(17):
        for _ in range(200):
            generator.yellow_short(region, char, True)

# green short
for char in range(37):
    for region in range(16):
        for _ in range(22):
            generator.green_short(region, char, True)
