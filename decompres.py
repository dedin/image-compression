from haaar import HuffmanCodesDict


def main():
    dictionary = HuffmanCodesDict
    f = open("output.bmp", "r+")
    compfile = f.read()
    print compfile
    print dictionary


def decode_hufman(dictionary, compfile):
    res = ""
    while huffmanCode:
        for k in dictionary:
            if huffmanCode.startswith(k):
                res += dictionary[k]
                huffmanCode = huffmanCode[len(k):]
    return res



def decode_quantization(pixelList, width, height):
    for i in range(height):
        for j in range(width):
            pixelList[i][j] = pixelList[i][j] * 100
    return pixelList


def decode_haar_transform():
    pass


main()