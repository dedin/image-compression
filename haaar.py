from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np
import math
import os


def main():
    picture = Image.open("smallbox.bmp")  # opening and reading the image header file, no access to image data yet
    pixels = picture.load()  # pixels is the access object(a 2d array) for the image
    coord = picture.size
    print "coord is ", coord
    print pixels[0, 0]  # No transparency values
    width, height = coord[0], coord[1]
    pixel_list = []
    # convert pixels tuple values and copy them into a 2d array
    for i in xrange(0, height):
        pixel_list.append([])
        for j in xrange(0, width):
            value = pixels[j, i]
            pixel_list[i].append((value[0] * 2 ^ 16) + (value[1] * 2 ^ 8) + (value[2]))

    # PAD LIST IF NECESSARY
    padded_pixel_list, new_width, new_height = pad_dimensions(pixel_list, width, height)
    # print "NEW WIDTH AND HEIGHT ARE {} AND {}".format(new_width, new_height)              #USE AN ASSERT STMT INSTEAD
    # DO HAAR TRANSFORM
    transform_coefficient = haar_transform(padded_pixel_list, new_width, new_height)
    pretty_print(transform_coefficient, "PIXEL VALUES AFTER HAAR GOING TO QUANTIZATION")
    # DO QUANTIZATION
    quantized_coefficient = quantize(transform_coefficient, new_width, new_height)
    pretty_print(quantized_coefficient, "PIXEL VALUES THAT WILL BE ENCODED USING HUFFMAN")
    # DO ENCODING
    root = variable_length_encode(quantized_coefficient, new_width, new_height)

    # START DECODING
    # HUFFMAN DECODE
    with open("output.bmp", "r") as enc_file:
        encoded_str = enc_file.read().replace("\n", "")
        print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
        pix_vals_to_dequant = decode_driver(encoded_str, root, new_height, new_width)

    pretty_print(pix_vals_to_dequant, "PIXEL VALUES AFTER HUFFMAN DECODING WAS DONE")

    # DEQUANTIZE
    de_haar = decode_quantization(pix_vals_to_dequant, new_width, new_height)

    # UNDO HAAR TRANSFORM


def haar_transform(pixelList, width, height):
    list_length = width * height
    print "LIST LENGTH IS ", list_length
    row_transformed_pixel_list = []
    # wavelet tranform for each row
    for row in pixelList:
        row_wavelet = wavelet_transform(row)
        row_transformed_pixel_list.append(row_wavelet)
    # pretty_print(row_transformed_pixel_list)
    # next do wavelet transform for each column
    new_arr = np.array(row_transformed_pixel_list)
    column_transformed = []
    for i in range(0, width):
        column_wavelet = wavelet_transform(new_arr[:, i])
        column_transformed.append(column_wavelet)
    # reshape the pixel array
    column_transformed_pixel_list = np.dstack(column_transformed)
    return column_transformed_pixel_list[0]


def decode_haar_transform():
    pass


def quantize(pixelList, width, height):
    for i in range(0, height):
        for j in range(0, width):
            if pixelList[i][j] < 0:
                pixelList[i][j] = (pixelList[i][j] // 100) + 1
            elif pixelList[i][j] == 0:
                pixelList[i][j] = pixelList[i][j]
            else:
                pixelList[i][j] = (pixelList[i][j] // 100)
    return pixelList


def decode_quantization(pixelList, width, height):
    for i in range(height):
        for j in range(width):
            pixelList[i][j] = pixelList[i][j] * 100
    return pixelList


def variable_length_encode(pixel_list, width, height):
    root = encode_driver(pixel_array=pixel_list, width=width, height=height)
    return root


def wavelet_transform(row):
    difference_list = []
    average_list, difference_list = haar_average(row, difference_list)
    while len(average_list) > 1:
        average_list, difference_list = haar_average(average_list, difference_list)
    difference_list.insert(0, average_list[0])
    assert (len(difference_list) == len(row))
    return difference_list


def haar_average(pixelList, diffList):
    avg_list = []
    for i in range(0, len(pixelList), 2):
        avg_list.append((pixelList[i] + pixelList[i + 1]) / 2)
        diffList.append((pixelList[i] - pixelList[i + 1]))
    return avg_list, diffList


def find_next_power(num):
    next_power = 1
    while next_power < num:
        next_power *= 2
    return next_power


def is_power(num):
    return ((num & (num - 1)) == 0) and num != 0


def pretty_print(pixelList, print_str):
    mx = max((len(str(pix_value)) for Line in pixelList for pix_value in Line))
    print "\n" * 2
    print print_str
    print "\n"
    for row in pixelList:
        print " ".join(["{:<{mx}}".format(Val, mx=mx) for Val in row])
    print "\n" * 6


def pad_dimensions(pixelList, width, height):
    # check if height is a power of 2
    changed = False
    WidthNextPower = width
    HeightNextPower = height
    if not is_power(height) or not is_power(width):
        changed = True
        if not is_power(height):
            # print"..ADJUSTING HEIGHT SIZE..."
            HeightNextPower = find_next_power(height)
            # make height padding by
            # - duplicating last row for all extra rows
            # - make new rows all of the same number ; the original last pixel (DOING THIS)
            # - Repeat each last pixel on each column
            HeightPadValue = pixelList[height - 1][width - 1]
            for i in range(height, HeightNextPower):
                pixelList.append([])
                for j in range(0, width):
                    pixelList[i].append(HeightPadValue)

        # check if number of pixels in width is a power of 2
        if not is_power(width):
            # print"...ADJUSTING WIDTH SIZE..."
            WidthNextPower = find_next_power(width)
            # pad each row with the last pixel value on that row
            for i in range(0, HeightNextPower):
                PadValue = pixelList[i][width - 1]
                for j in range(width, (WidthNextPower)):
                    pixelList[i].append(PadValue)

    if changed is True:
        return pixelList, HeightNextPower, WidthNextPower
    else:
        return pixelList, HeightNextPower, WidthNextPower


def normalized_haar_transform(pixelList):
    n = len(pixelList)
    root = int(math.sqrt(n))
    for i in range(0, n):
        pixelList[i] = pixelList[i] / root
    while n >= 2:
        pixelList = nwt_step(pixelList, n)
        n = int(n / 2)
    return pixelList


def nwt_step(pixels, j):
    index = int(j / 2)
    root = math.sqrt(2)
    b = [0.0 for i in pixels]
    for i in range(0, index):
        b[i] = (pixels[(2 * i) - 1] + pixels[2 * i]) / root
        b[int(j / 2 + i)] = (pixels[(2 * i) - 1] - pixels[2 * i]) / root
    return b


main()
