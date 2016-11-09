from __future__ import division
from PIL import Image
from hufman import encode_driver, decode_driver
import numpy as np
import math
import os


def main():
    Picture = Image.open("smallbox.bmp")  # just opening and reading the image header file, no access to image data yet
    Pixels = Picture.load()  # pixels is the access object(a 2d array) for the image
    Coord = Picture.size
    print "coord is ", Coord
    print Pixels[0, 0]  # No transparency values
    Width, Height = Coord[0], Coord[1]
    PixelList = []
    # convert pixels tuple values and copy them into a 2d array
    for i in xrange(0, Height):
        PixelList.append([])
        for j in xrange(0, Width):
            value = Pixels[j, i]
            PixelList[i].append((value[0] * 2 ^ 16) + (value[1] * 2 ^ 8) + (value[2]))

    # PAD LIST IF NECESSARY
    PaddedPixelList, NewWidth, NewHeight = pad_dimensions(PixelList, Width, Height)
    # print "NEW WIDTH AND HEIGHT ARE {} AND {}".format(NewWidth, NewHeight)              #USE AN ASSERT STMT INSTEAD
    #DO HAAR TRANSFORM
    TransformCoefficient = haar_transform(PaddedPixelList, NewWidth, NewHeight)
    pretty_print(TransformCoefficient, "PIXEL VALUES AFTER HAAR GOING TO QUANTIZATION")
    #DO QUANTIZATION
    QuantizedCoefficient =quantize(TransformCoefficient, NewWidth, NewHeight)
    pretty_print(QuantizedCoefficient, "PIXEL VALUES THAT WILL BE ENCODED USING HUFFMAN")
    # DO ENCODING
    root = variable_length_encode(QuantizedCoefficient, NewWidth, NewHeight)

    #START DECODING
    # HUFFMAN DECODE
    with open("output.bmp", "r") as enc_file:
        encoded_str = enc_file.read().replace("\n", "")
        print "SIZE OF STRING TO DECODE IS ", len(encoded_str)
        pix_vals_to_dequant = decode_driver(encoded_str , root, NewHeight, NewWidth)

    pretty_print(pix_vals_to_dequant, "PIXEL VALUES AFTER HUFFMAN DECODING WAS DONE")

    # DEQUANTIZE
    de_haar = decode_quantization(pix_vals_to_dequant, NewWidth, NewHeight)

    # UNDO HAAR TRANSFORM



def haar_transform(pixelList, Width, Height):
    ListLength = Width * Height
    print "LIST LENGTH IS ", ListLength
    RowTransformedPixelList = []
    #wavelet tranform for each row
    for row in pixelList:
        RowWavelet = wavelet_transform(row)
        RowTransformedPixelList.append(RowWavelet)
    # pretty_print(RowTransformedPixelList)
    #next do wavelet transform for each column
    NewArr = np.array(RowTransformedPixelList)
    ColumnTransformed = []
    for i in range(0,Width):
        ColumnWavelet = wavelet_transform(NewArr[:,i])
        ColumnTransformed.append(ColumnWavelet)
    #reshape the pixel array
    ColumnTransformedPixelList = np.dstack(ColumnTransformed)
    return ColumnTransformedPixelList[0]



def quantize(pixelList,width, height):
    for i in range(0, height):
        for j in range(0,width):
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
    root = encode_driver(pixel_array=pixel_list, width=width, height= height)
    return root



def wavelet_transform(row):
    DifferenceList = []
    AverageList, DifferenceList = haar_average(row, DifferenceList)
    while len(AverageList) > 1:
        AverageList, DifferenceList = haar_average(AverageList, DifferenceList)
    DifferenceList.insert(0,AverageList[0])
    assert (len(DifferenceList) == len(row))
    return DifferenceList


def haar_average(pixelList, diffList):
    AvgList = []
    for i in range(0, len(pixelList), 2):
        AvgList.append((pixelList[i] + pixelList[i + 1]) / 2)
        diffList.append((pixelList[i] - pixelList[i + 1]))
    return AvgList, diffList


def find_next_power(num):
    NextPower = 1
    while NextPower < num:
        NextPower *= 2
    return NextPower


def is_power(num):
    return ((num & (num - 1)) == 0) and num != 0




def pretty_print(pixelList, print_str):
    mx = max((len(str(PixValue)) for Line in pixelList for PixValue in Line))
    print "\n" * 2
    print print_str
    print "\n"
    for row in pixelList:
        print " ".join(["{:<{mx}}".format(Val, mx=mx) for Val in row])
    print "\n" * 6


def pad_dimensions(pixelList, Width, Height):
    # check if height is a power of 2
    changed = False
    WidthNextPower = Width
    HeightNextPower = Height
    if not is_power(Height) or not is_power(Width):
        changed = True
        if not is_power(Height):
            # print"..ADJUSTING HEIGHT SIZE..."
            HeightNextPower = find_next_power(Height)
            # make height padding by
            # - duplicating last row for all extra rows
            # - make new rows all of the same number ; the original last pixel (DOING THIS)
            # - Repeat each last pixel on each column
            HeightPadValue = pixelList[Height - 1][Width - 1]
            for i in range(Height, HeightNextPower):
                pixelList.append([])
                for j in range(0, Width):
                    pixelList[i].append(HeightPadValue)
            # print"CHANGED HEIGHT FROM {} TO {}".format(Height, HeightNextPower)

        # check if number of pixels in width is a power of 2
        if not is_power(Width):
            # print"...ADJUSTING WIDTH SIZE..."
            WidthNextPower = find_next_power(Width)
            # pad each row with the last pixel value on that row
            for i in range(0, HeightNextPower):
                PadValue = pixelList[i][Width - 1]
                for j in range(Width, (WidthNextPower)):
                    pixelList[i].append(PadValue)
            # print"CHANGED WIDTH FROM {} TO {}".format(Width, WidthNextPower)


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
        pixelList = NWTstep(pixelList, n)
        n = int(n / 2)
    return pixelList


def NWTstep(pixels, j):
    index = int(j / 2)
    root = math.sqrt(2)
    b = [0.0 for i in pixels]
    for i in range(0, index):
        b[i] = (pixels[(2 * i) - 1] + pixels[2 * i]) / root
        b[int(j / 2 + i)] = (pixels[(2 * i) - 1] - pixels[2 * i]) / root
    return b


main()

# DifferenceList = []
# AverageList, DifferenceList = haar_average(pixelList, DifferenceList)
# while len(AverageList) > 1:
#     AverageList, DifferenceList = haar_average(AverageList, DifferenceList)
#
# DifferenceList.insert(0, AverageList[0])
# assert (len(DifferenceList) == len(pixelList))
# return DifferenceList












