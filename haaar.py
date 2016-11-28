from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np
import math


def main():
    picture = Image.open("smallbox.bmp")  # opening and reading the image header file, no access to image data yet
    pixels = picture.load()  # pixels is the access object(a 2d array) for the image
    coord = picture.


    r
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


    l = [[20,20,30,40],[30,20,30,40],[40,30,20,30],[30,20,10,40]]
    w = 4
    h = 4
    # transform_coefficient = haar_transform(padded_pixel_list, new_width, new_height)
    transform_coefficient = haar_transform(l, w, h)
    pretty_print(transform_coefficient, "PIXEL VALUES AFTER HAAR GOING TO QUANTIZATION")

    # # DO QUANTIZATION
    # quantized_coefficient = quantize(transform_coefficient, new_width, new_height)
    # pretty_print(quantized_coefficient, "PIXEL VALUES THAT WILL BE ENCODED USING HUFFMAN")
    # # DO ENCODING
    # root = variable_length_encode(quantized_coefficient, new_width, new_height)
    #
    # # START DECODING
    # # HUFFMAN DECODE
    # with open("output.bmp", "r") as enc_file:
    #     encoded_str = enc_file.read().replace("\n", "")
    #     print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
    #     pix_vals_to_dequant = decode_driver(encoded_str, root, new_height, new_width)
    #
    # pretty_print(pix_vals_to_dequant, "PIXEL VALUES AFTER HUFFMAN DECODING WAS DONE")
    #
    # # DEQUANTIZE
    # de_haar = decode_quantization(pix_vals_to_dequant, new_width, new_height)

    # UNDO HAAR TRANSFORM


def haar_transform(pixel_list, width, height):
    list_length = width * height
    print "LIST LENGTH IS ", list_length

    pretty_print(pixel_list, "beginning list")
    row_transformed_pixel_list = []
    # wavelet transform for each row
    for row in pixel_list:
        row_wavelet = wavelet_transform(row)
        row_transformed_pixel_list.append(row_wavelet)
    pretty_print(row_transformed_pixel_list, "After row tranform")
    # pretty_print(row_transformed_pixel_list)
    # next do wavelet transform for each column
    new_arr = np.array(row_transformed_pixel_list)
    pretty_print(new_arr, "FLIPPED ARRAY")
    column_transformed = []
    for i in range(0, width):
        column_wavelet = wavelet_transform(new_arr[:, i])
        column_transformed.append(column_wavelet)
    pretty_print(column_transformed, "After column transform")
    # reshape the pixel array
    column_transformed_pixel_list = np.dstack(column_transformed)
    return column_transformed_pixel_list[0]


def decode_haar_transform():
    pass


def quantize(pixel_list, width, height):
    for i in range(0, height):
        for j in range(0, width):
            if pixel_list[i][j] < 0:
                pixel_list[i][j] = (pixel_list[i][j] // 100) + 1
            elif pixel_list[i][j] == 0:
                pixel_list[i][j] = pixel_list[i][j]
            else:
                pixel_list[i][j] = (pixel_list[i][j] // 100)
    return pixel_list


def decode_quantization(pixel_list, width, height):
    for i in range(height):
        for j in range(width):
            pixel_list[i][j] *= 100
    return pixel_list


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


def haar_average(pixel_list, diff_list):
    avg_list = []
    list_length = len(pixel_list)
    j = 1
    for average_index in range(0, len(pixel_list), 2):
        avg_list.append((pixel_list[average_index] + pixel_list[average_index + 1]) / 2)
        difference_index = list_length - (2*j)
        diff_list.insert(0, (pixel_list[difference_index] - pixel_list[difference_index + 1]))
        j += 1
    return avg_list, diff_list


def de_code(decoded_row):
    """"Converts from a row of one average and multiple differences to the original values

    Parameters
    ----------
       :param decoded_row:
       A list containing an average followed by a list of differences.  The total size must be a power of 2.
    """
    average_list = []
    width = len(decoded_row)
    diff = decoded_row[1] / 2
    average_list.append(decoded_row[0] + diff)
    average_list.append(decoded_row[0] - diff)
    diff_list = list(decoded_row[2:])
    while len(average_list) < width:
        index = 0
        average_list = d_code(average_list, diff_list, (2**index) - 1)
        index += 1
    return average_list


def d_code(average_list, diff_list, diff_list_index):
    avg_list = []
    for avg in average_list:
        diff_avg = diff_list[diff_list_index] / 2
        avg_list.append(avg + diff_avg)
        avg_list.append(avg - diff_avg)
        diff_list_index += 1
    return avg_list


def find_next_power(num):
    # import math
    # return 2 ** (floor(math.log(num, 2)) + 1)

    next_power = 1
    while next_power < num:
        next_power *= 2
    return next_power


def is_power(num):
    return ((num & (num - 1)) == 0) and num != 0


def pretty_print(pixel_list, print_str):
    mx = max((len(str(pix_value)) for Line in pixel_list for pix_value in Line))
    print "\n" * 2
    print print_str
    print "\n"
    for row in pixel_list:
        print " ".join(["{:<{mx}}".format(Val, mx=mx) for Val in row])
    print "\n" * 6


def pad_dimensions(pixel_list, width, height):
    # check if height is a power of 2
    changed = False
    width_next_power = width
    height_next_power = height
    if not is_power(height) or not is_power(width):
        changed = True
        if not is_power(height):
            # print"..ADJUSTING HEIGHT SIZE..."
            height_next_power = find_next_power(height)
            # make height padding by
            # - duplicating last row for all extra rows
            # - make new rows all of the same number ; the original last pixel (DOING THIS)
            # - Repeat each last pixel on each column
            height_pad_value = pixel_list[height - 1][width - 1]
            for i in range(height, height_next_power):
                pixel_list.append([])
                for j in range(0, width):
                    pixel_list[i].append(height_pad_value)

        # check if number of pixels in width is a power of 2
        if not is_power(width):
            # print"...ADJUSTING WIDTH SIZE..."
            width_next_power = find_next_power(width)
            # pad each row with the last pixel value on that row
            for i in range(0, height_next_power):
                pad_value = pixel_list[i][width - 1]
                for j in range(width, width_next_power):
                    pixel_list[i].append(pad_value)

    if changed is True:
        return pixel_list, height_next_power, width_next_power
    else:
        return pixel_list, height_next_power, width_next_power


def normalized_haar_transform(pixel_list):
    n = len(pixel_list)
    root = int(math.sqrt(n))
    for i in range(0, n):
        pixel_list[i] /= root
    while n >= 2:
        pixel_list = nwt_step(pixel_list, n)
        n = int(n / 2)
    return pixel_list


def nwt_step(pixels, j):
    index = int(j / 2)
    root = math.sqrt(2)
    b = [0.0 for i in pixels]
    for i in range(0, index):
        b[i] = (pixels[(2 * i) - 1] + pixels[2 * i]) / root
        b[int(j / 2 + i)] = (pixels[(2 * i) - 1] - pixels[2 * i]) / root
    return b


main()
