from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np
import math



# TODO
# Everything works fine. clean up comments and make variable names better, push to git hub
# Show image from pixel values
#put try catch around file opens


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



def main():
    picture = Image.open("smallbox.bmp")
    pixels = picture.load()
    coord = picture.size
    print "coord is ", coord
    print pixels[0, 39]
    width, height = coord[0], coord[1]                  #check this again!!!!!!! the height and width
    pixel_list = []
    # convert pixels tuple values and copy them into a 2d array
    for i in xrange(0, height):
        pixel_list.append([])
        for j in xrange(0, width):
            value = pixels[j, i]
            pixel_list[i].append(rgb_to_num(value))


    # pretty_print(pixel_list, "at the beginning")

    # PAD LIST IF NECESSARY
    padded_pixel_list, new_width, new_height = pad_dimensions(pixel_list, width, height)
    # print "NEW WIDTH AND HEIGHT ARE {} AND {}".format(new_width, new_height)              #USE AN ASSERT STMT INSTEAD

    # DO HAAR TRANSFORM
    transform_coefficient = haar_transform(padded_pixel_list, new_width, new_height)
    # pretty_print(transform_coefficient, "PIXEL VALUES AFTER HAAR GOING TO QUANTIZATION")



    # DO QUANTIZATION

    quantized_coefficient = quantize(transform_coefficient, new_width, new_height)
    # pretty_print(quantized_coefficient, "PIXEL VALUES THAT WILL BE ENCODED USING HUFFMAN")


    # DO ENCODING
    root = variable_length_encode(quantized_coefficient, new_width, new_height)


    # START DECODING

    # HUFFMAN DECODE
    with open("encodeoutput.bmp", "r") as enc_file:
        encoded_str = enc_file.read()
        print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
        pix_vals_to_dequant = decode_driver(encoded_str, root, new_height, new_width)

    # pretty_print(pix_vals_to_dequant, "PIXEL VALUES AFTER HUFFMAN DECODING WAS DONE")



    # DEQUANTIZE
    de_haar = decode_quantization(pix_vals_to_dequant, new_width, new_height)
    # pretty_print(de_haar, "AFTER DEQUANTIZE")


    # UNDO HAAR TRANSFORM
    decoded_pixel_value = decode_haar_transform(de_haar, new_width, new_height)
    # pretty_print(decoded_pixel_values, "AFTER ALL DECODING")

    # REMOVE PADDING
    decoded_pixel_values = remove_pad(decoded_pixel_value, width, height)

    # GET RGB VALUES BACK FROM INTEGERS
    rgb_list = []
    for i in range(height):
        rgb_list.append([])
        for j in range(width):
            rgb_list[i].append(num_to_rgb(decoded_pixel_values[i][j]))
    # pretty_print(rgb_array, "")

    # MAKE IMAGE FILE
    img_array = np.array(rgb_list, np.uint8)
    pilimage = Image.fromarray(img_array)
    pilimage.save('out.bmp')


def remove_pad(pixel_list, width, height):
    new_list = []
    for i in range(0, height):
        new_list.append([])
        for j in range(0, width):
            new_list[i].append(pixel_list[i][j])
    return new_list

def haar_transform(pixel_list, width, height):
    list_length = width * height
    print "LIST LENGTH IS ", list_length

    # pretty_print(pixel_list, "beginning list")
    row_transformed_pixel_list = []
    # wavelet transform for each row
    for row in pixel_list:
        row_wavelet = wavelet_transform(row)
        row_transformed_pixel_list.append(row_wavelet)
    #pretty_print(row_transformed_pixel_list, "After row tranform")
    # next do wavelet transform for each column
    new_arr = np.array(row_transformed_pixel_list)
    # pretty_print(new_arr, "FLIPPED ARRAY")
    column_transformed = []
    for i in range(0, width):
        # print "ENCODING  AND I SHOULD GET", new_arr[:, i]
        column_wavelet = wavelet_transform(new_arr[:, i])
        # print "AFTER " ,column_wavelet
        # print "\n"
        column_transformed.append(column_wavelet)
    #pretty_print(column_transformed, "After column transform BEFORE REARRANGE")
    # reshape the pixel array
    column_transformed_pixel_list = np.dstack(column_transformed)
    #pretty_print(column_transformed_pixel_list[0], "REARRANGING AND RETURNING")
    return column_transformed_pixel_list[0]


def decode_haar_transform(coded_list, width, height):
    #pretty_print(coded_list, "WHAT I GOT")
    col_re_arranged_to_row = np.array(coded_list)
    array_2 = []
    for i in range(0, width):
        #print "going to decode ", col_re_arranged_to_row[:, i]
        decoded_row = de_code(col_re_arranged_to_row[:, i])
        #print "Got back ", decoded_row
        array_2.append(decoded_row)
    row_re_arranged_to_col = np.dstack(array_2)
    final_decode = []
    #print "\n" * 3
    for row in row_re_arranged_to_col[0]:
        #print "THIS IS THE ROW TO DECODE ", row
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
        # diff_avg = diff_list[diff_list_index] / 2
        diff_avg = diff_list[0] / 2
        avg_list.append(avg + diff_avg)
        avg_list.append(avg - diff_avg)
        # diff_list_index += 1
        diff_list.pop(0)
    return avg_list, diff_list



con = 2

def decode_quantization(pixelList, width, height):
    for i in range(height):
        for j in range(width):
            if pixelList[i][j] < 0:
                pixelList[i][j] = pixelList[i][j] * con
            elif pixelList[i][j] == 0:
                pixelList[i][j] = 0
            else:
                pixelList[i][j] = pixelList[i][j] * con
    return pixelList



def quantize(pixelList, width, height):
    for i in range(0, height):
        for j in range(0, width):
            pixelList[i][j] = pixelList[i][j] // con
    return pixelList

# def quantize(pixelList, width, height):
#     for i in range(0, height):
#         for j in range(0, width):
#             if pixelList[i][j] < 0:
#                 pixelList[i][j] = (pixelList[i][j] // con) + 1
#             elif pixelList[i][j] == 0:
#                 pixelList[i][j] = pixelList[i][j]
#             else:
#                 pixelList[i][j] = (pixelList[i][j] // con)
#     return pixelList
#
#
# def decode_quantization(pixelList, width, height):
#     for i in range(height):
#         for j in range(width):
#             pixelList[i][j] = pixelList[i][j] * con
#     return pixelList


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

#
# def haar_average(pixelList, diffList):
#     avg_list = []
#     for i in range(0, len(pixelList), 2):
#         avg_list.append((pixelList[i] + pixelList[i + 1]) / 2)
#         diffList.append((pixelList[i] - pixelList[i + 1])) # insert instead
#     return avg_list, diffList



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
