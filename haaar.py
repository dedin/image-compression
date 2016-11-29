from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np
import math


# TODO
# trim padded area
# put try catch around file opens


def main():
    picture = Image.open("smallbox.bmp")
    pixels = picture.load()
    coord = picture.size
    print "coord is ", coord
    pixel_size = len(pixels[0, 0])
    print pixels[6, 6][0]
    print pixels[0,0][1]
    width, height = coord[0], coord[1]  # check this again!!!!!!! the height and width
    color_list = []

    # convert pixels tuple values and copy them into a 2d array
    for x in xrange(0, pixel_size):
        pixel_list = []
        for i in xrange(0, height):
            pixel_list.append([])
            for j in xrange(0, width):
                value = pixels[j, i][x]
                pixel_list[i].append(value)
        quantized_pix = process_col(pixel_list, width, height)
        color_list.append(quantized_pix)

    # convert values into one for each pixel
    assert len(color_list) is 3
    for i in range(0, 3):
        pass




def process_col(pixel_list, width, height):
    # pretty_print(pixel_list, "AT THE BEGINNING")

    # PAD LIST IF NECESSARY
    padded_pixel_list, new_width, new_height = pad_dimensions(pixel_list, width, height)
    assert new_width == new_height

    # HAAR TRANSFORM
    transform_coefficient = haar_transform(padded_pixel_list, new_width, new_height)

    # QUANTIZATION
    quantized_coefficient = quantize(transform_coefficient, new_width, new_height)

    return quantized_coefficient

    # # HUFFMAN ENCODE
    # root = variable_length_encode(quantized_coefficient, new_width, new_height)

    # # HUFFMAN DECODE
    # with open("encodeoutput.bmp", "r") as enc_file:
    #     encoded_str = enc_file.read()
    #     print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
    #     pix_val_to_dequant = decode_driver(encoded_str, root, new_height, new_width)
    #
    # # DE QUANTIZE
    # de_haar = decode_quantization(pix_val_to_dequant, new_width, new_height)
    #
    # # HAAR TRANSFORM DECODE
    # decoded_pixel_values = decode_haar_transform(de_haar, new_width, new_height)
    #
    # # SHOW IMAGE FROM DECODED INTEGERS
    # rgb_list = []
    # for i in range(new_height):
    #     rgb_list.append([])
    #     for j in range(new_width):
    #         rgb_list[i].append(num_to_rgb(decoded_pixel_values[i][j]))
    # img_array = np.array(rgb_list, np.uint8)
    # pil_image = Image.fromarray(img_array)
    # pil_image.save('out.bmp')


def haar_transform(pixel_list, width, height):
    list_length = width * height
    print "LIST LENGTH IS ", list_length

    #pretty_print(pixel_list, "beginning list")
    row_transformed_pixel_list = []
    # wavelet transform for each row
    for row in pixel_list:
        row_wavelet = wavelet_transform(row)
        row_transformed_pixel_list.append(row_wavelet)
    # wavelet transform for each column
    new_arr = np.array(row_transformed_pixel_list)
    column_transformed = []
    for i in range(0, width):
        column_wavelet = wavelet_transform(new_arr[:, i])
        column_transformed.append(column_wavelet)
    # reshape the pixel array
    column_transformed_pixel_list = np.dstack(column_transformed)
    return column_transformed_pixel_list[0]


def decode_haar_transform(coded_list, width, height):
    col_re_arranged_to_row = np.array(coded_list)
    array_2 = []
    for i in range(0, width):
        decoded_row = de_code(col_re_arranged_to_row[:, i])
        array_2.append(decoded_row)
    row_re_arranged_to_col = np.dstack(array_2)
    final_decode = []
    for row in row_re_arranged_to_col[0]:
        final_decode.append(de_code(row))
    return final_decode


def de_code(row_to_decode):
    """"Converts from a row of one average and multiple differences to the original values
    Parameters
    ----------
       :param row_to_decode:
       A list containing an average followed by a list of differences.  The total size must be a power of 2.
    """
    average_list = []
    width = len(row_to_decode)
    diff = row_to_decode[1] / 2
    average_list.append(row_to_decode[0] + diff)
    average_list.append(row_to_decode[0] - diff)
    diff_list = list(row_to_decode[2:])
    index = 0
    while len(average_list) < width:
        diff_list_index = (2 ** index) - 1
        average_list, diff_list = d_code(average_list, diff_list, diff_list_index)
        index += 1
    return average_list


def d_code(average_list, diff_list, diff_list_index):
    avg_list = []
    for avg in average_list:
        diff_avg = diff_list[0] / 2
        avg_list.append(avg + diff_avg)
        avg_list.append(avg - diff_avg)
        diff_list.pop(0)
    return avg_list, diff_list


con = 2


def quantize(pixel_list, width, height):
    for i in range(0, height):
        for j in range(0, width):
            if pixel_list[i][j] < 0:
                pixel_list[i][j] = (pixel_list[i][j] // con) + 1
            elif pixel_list[i][j] == 0:
                pixel_list[i][j] = pixel_list[i][j]
            else:
                pixel_list[i][j] = (pixel_list[i][j] // con)
    return pixel_list


def decode_quantization(pixel_list, width, height):
    for i in range(height):
        for j in range(width):
            pixel_list[i][j] = pixel_list[i][j] * con
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
        difference_index = list_length - (2 * j)
        diff_list.insert(0, (pixel_list[difference_index] - pixel_list[difference_index + 1]))
        j += 1
    return avg_list, diff_list


def find_next_power(num):
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
    changed = False
    width_next_power = width
    height_next_power = height
    if not is_power(height) or not is_power(width):
        changed = True
        if not is_power(height):
            height_next_power = find_next_power(height)
            height_pad_value = pixel_list[height - 1][width - 1]
            for i in range(height, height_next_power):
                pixel_list.append([])
                for j in range(0, width):
                    pixel_list[i].append(height_pad_value)
        if not is_power(width):
            width_next_power = find_next_power(width)
            for i in range(0, height_next_power):
                pad_value = pixel_list[i][width - 1]
                for j in range(width, (width_next_power)):
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


def rgb_to_num(rgb_value):
    red = rgb_value[0]
    green = rgb_value[1]
    blue = rgb_value[2]
    rgb_num = (red << 16) + (green << 8) + blue
    return rgb_num


def num_to_rgb(rgb_num):
    red = (int(rgb_num) >> 16) & 255
    green = (int(rgb_num) >> 8) & 255
    blue = int(rgb_num) & 255
    return red, green, blue

main()
