from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np


QUANTIZATION_CON = 50
NUMBER_OF_BITS_TO_SHIFT = 9
NUMBER_MASK = (1 << NUMBER_OF_BITS_TO_SHIFT) - 1
NEGATIVE_RANGE = 1 << (NUMBER_OF_BITS_TO_SHIFT - 1)


def main():
    picture = Image.open("bird.jpeg")
    pixels = picture.load()
    coord = picture.size
    print "coord is ", coord
    pixel_size = len(pixels[0, 0])
    width, height = coord[0], coord[1]
    color_list = []
    data_list = []
    for i in xrange(0, height):
        data_list.append([])
        for j in xrange(0, width):
            value = pixels[j, i]
            data_list[i].append(value)
    # Extract each color component from each pixel into separate arrays
    # process each component array - do haar transform and quantize
    for x in xrange(0, pixel_size):
        pixel_list = []
        for i in xrange(0, height):
            pixel_list.append([])
            for j in xrange(0, width):
                value = pixels[j, i][x]
                pixel_list[i].append(value)
        quantized_pix = process_component(pixel_list, width, height)
        color_list.append(quantized_pix)
    # combine pixel components back into one for each pixel and encode to file
    assert len(color_list) is 3
    new_height = len(color_list[0])
    new_width = len(color_list[0][0])
    val_to_encode = []
    rgb_list = []
    for i in range(0, new_height):
        val_to_encode.append([])
        for j in range(0, new_width):
            rgb_list.append(color_list[0][i][j])
            rgb_list.append(color_list[1][i][j])
            rgb_list.append(color_list[2][i][j])
            val_to_encode[i].append(rgb_to_num(rgb_list))
            rgb_list = []
    root = variable_length_encode(val_to_encode, new_width, new_height)
    # DECODE file back to pixel values
    decode_file(root, new_width, new_height, width, height)


# Function to decode the file back to pixel values
# Parameters : root of the huffman tree
#              width and height before padding and after padding
def decode_file(root, width, height, old_width, old_height):
    with open("encodeoutput.bmp", "r") as enc_file:
        encoded_str = enc_file.read()
        print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
        pix_val_to_dequant = decode_driver(encoded_str, root, height, width)
    red_list, green_list, blue_list = make_lists(pix_val_to_dequant)
    new_red_list = remove_pad(dequant_dehaar(red_list), old_width, old_height)
    new_green_list = remove_pad(dequant_dehaar(green_list), old_width, old_height)
    new_blue_list = remove_pad(dequant_dehaar(blue_list), old_width, old_height)
    rgb_list = make_pixel_arr(new_red_list, new_green_list, new_blue_list)
    # Show the Image from pixel values
    img_array = np.array(rgb_list, np.uint8)
    pil_image = Image.fromarray(img_array)
    pil_image.save('out.bmp')


# Function to run Haar transform and quantization on each component array
# Parameters : each component array, the width and height pf the array
# Returns a quantized and Haar transformed pixel array
def process_component(pixel_list, width, height):
    padded_pixel_list, new_width, new_height = pad_if_necessary(pixel_list, width, height)
    print new_width, new_height
    assert new_width == new_height
    transform_coefficient = haar_transform(padded_pixel_list, new_width, new_height)
    quantized_coefficient = quantize(transform_coefficient, new_width, new_height)
    return quantized_coefficient


# Function to combine the 3 component arrays into one pixel array
# parameters : Each component array of red, gree, and blue commponent
# Returns a combined array of pixel values
def make_pixel_arr(red_list, green_list, blue_list):
    height = len(red_list)
    width = len(red_list[0])
    rgb_list = []
    for i in range(height):
        rgb_list.append([])
        for j in range(width):
            rgb_list[i].append(make_set(red_list, green_list, blue_list, i, j))
    return rgb_list


# Helper function to make a python set of pixel values
# Used by make_pixel_arr()
def make_set(red_list, green_list, blue_list, i, j):
    red = check_range(int(red_list[i][j]))
    green = check_range(int(green_list[i][j]))
    blue = check_range(int(blue_list[i][j]))
    return red, green, blue


# Function to check and adjust the value of a pixel
def check_range(val):
    if val > 255:
        val = 255
    elif val < 0:
        val = 0
    return val


# Function to perform decoding Haar transform and quantization
# Parameters : A 2 dimensional array which is essentially the pixel array for
#              each color component
# Returns decoded pixel values
def dequant_dehaar(col_list):
    height = len(col_list)
    width = len(col_list[0])
    # DE QUANTIZE
    de_haar = decode_quantization(col_list, width, height)
    # HAAR TRANSFORM DECODE
    decoded_pixel_values = decode_haar_transform(de_haar, width, height)
    return decoded_pixel_values


# Function to split the huffman decoded values into the 3 color components
# arrays of red, green and blue.
# Parameters : Huffman decoded array of values
# Returns 3 component array of red, green and blue
def make_lists(decoded_pix_list):
    height = len(decoded_pix_list)
    width = len(decoded_pix_list[0])
    red_list = []
    green_list = []
    blue_list = []
    for i in range(0, height):
        red_list.append([])
        green_list.append([])
        blue_list.append([])
        for j in range(0, width):
            val = decoded_pix_list[i][j]
            color_set = num_to_rgb(val)
            red_list[i].append(color_set[0])
            green_list[i].append(color_set[1])
            blue_list[i].append(color_set[2])
    return red_list, green_list, blue_list


# Function to remove the padding applied to the pixel array
# before Haar transform
# Parameters : A 2 dimensional array of numbers corresponding to
#              each color component array, the width and the height
# return a 2 dimensional array where pad has been removed if necessary
def remove_pad(pixel_list, width, height):
    new_list = []
    for i in range(0, height):
        new_list.append([])
        for j in range(0, width):
            new_list[i].append(pixel_list[i][j])
    return new_list


# Function to perform Haar transform on a 2 dimenional pixel array
# Parameters : A 2 dimensional array of pixel values that corresponds to
#             each color component array, the width and the height
# Returns a 2 dimensional array of transform coefficients
def haar_transform(pixel_list, width, height):
    list_length = width * height
    print "LIST LENGTH IS ", list_length
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


# Function to decode the Haar transform
# Parameters : A 2 dimensional array of de-quantized values corresponding
#              to each color component, the width and the height
# Returns the resulting 2 dimensional array that is a result of decoding
# Haar transform
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


# Helper Function for decode_haar_transform() to convert a row or column at
# of one average and multiple differences to the original values
# Parameters : A list of average and differences. Total size must be a power of 2
# Returns a decoded list of values
def de_code(row_to_decode):
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


# Helper function for de_code()
def d_code(average_list, diff_list, diff_list_index):
    avg_list = []
    for avg in average_list:
        diff_avg = diff_list[0] / 2
        avg_list.append(avg + diff_avg)
        avg_list.append(avg - diff_avg)
        diff_list.pop(0)
    return avg_list, diff_list


# Function to perform quantization on each pixel array
# Parameters : A 2 dimensional array of pixel array corresponding to
#              each color component, the width and the height
# Returns a quantized 2 dimensional array
def quantize(pixel_list, width, height):
    for i in range(0, height):
        for j in range(0, width):
            if pixel_list[i][j] < 0:
                pixel_list[i][j] = (pixel_list[i][j] // QUANTIZATION_CON) + 1
            elif pixel_list[i][j] == 0:
                pixel_list[i][j] = pixel_list[i][j]
            else:
                pixel_list[i][j] = (pixel_list[i][j] // QUANTIZATION_CON)
    return pixel_list


# Function to decode quantization in each pixel array of color components
# Parameters : A 2 dimensional array of values, the width and the height
# Returns a 2 dimensional array of de-quantized values
def decode_quantization(pixel_list, width, height):
    for i in range(height):
        for j in range(width):
            pixel_list[i][j] = pixel_list[i][j] * QUANTIZATION_CON
    return pixel_list


# Function to perform variable length encoding of 2 dimensional pixel array to file using Huffman
# Parameters : A 2 dimensional array of values to be encoded to file.
# Returns root from the Huffman tree
def variable_length_encode(pixel_list, width, height):
    root = encode_driver(pixel_array=pixel_list, width=width, height=height)
    return root


# Helper function to perform wavelet transform of each row or column
# Parameter : The row or column to perform wavelet transform on
# Returns a list of transform coefficients
def wavelet_transform(row):
    difference_list = []
    average_list, difference_list = haar_average(row, difference_list)
    while len(average_list) > 1:
        average_list, difference_list = haar_average(average_list, difference_list)
    difference_list.insert(0, average_list[0])
    assert (len(difference_list) == len(row))
    return difference_list


# helper function for wavelet_transform() to get one average followed by set of
# differences for a row or column
# Parameters : A list corresponding to one average 
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


def pad_if_necessary(pixel_list, width, height):
    print "PADDING"
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


def convert_neg_number_if_needed(number):
    if number < 0:
        number = NEGATIVE_RANGE + abs(number)
    return number


def convert_neg_back_if_needed(number):
    if number > NEGATIVE_RANGE:
        number = -(number - NEGATIVE_RANGE)
    return number


def rgb_to_num(rgb_value):
    red = int(convert_neg_number_if_needed(rgb_value[0]))
    green = int(convert_neg_number_if_needed(rgb_value[1]))
    blue = int(convert_neg_number_if_needed(rgb_value[2]))
    rgb_num = (red << (NUMBER_OF_BITS_TO_SHIFT * 2)) | (green << NUMBER_OF_BITS_TO_SHIFT) | blue
    return rgb_num


def num_to_rgb(rgb_num):
    red = convert_neg_back_if_needed((int(rgb_num) >> (NUMBER_OF_BITS_TO_SHIFT * 2)) & NUMBER_MASK)

    green = convert_neg_back_if_needed((int(rgb_num) >> NUMBER_OF_BITS_TO_SHIFT) & NUMBER_MASK)

    blue = convert_neg_back_if_needed(int(rgb_num) & NUMBER_MASK)

    return red, green, blue


if __name__ == "__main__":
    main()

