

# Was in haar.py before, used to make the frequencies and then encode
def variable_length_encode(pixelList, width, height):
    PixelFrequencies = {}
    #get the table of frequencies
    for i in range(height):
        for j in range(width):
            key = pixelList[i][j]
            if key in PixelFrequencies:
                PixelFrequencies[key] += 1
            else:
                PixelFrequencies[key] = 1
    huffmanEncode(PixelFrequencies)



# Was in haar.py before, build the tree and calls get_codes to
# make the huffman codes
def huffmanEncode(frequencyList):
    NodeList = []
    for key, value in frequencyList.iteritems():
        NodeList.append(Node(key,value))
    heapify(NodeList)
    while len(NodeList) > 1:
        LeftChild = heappop(NodeList)
        RightChild = heappop(NodeList)
        NewNode = Node(None, (LeftChild.freq + RightChild.freq))
        NewNode.set_children(LeftChild, RightChild)
        heappush(NodeList, NewNode)

    get_codes("", NodeList[0])

# Traverses the tree to get the huffman codes and uses the global
# dictionary to make a map of pixel value to huffman code
def get_codes(prevStr, Node):
    if Node.value is not None:              # if its a leaf
        if prevStr is None:
            HuffmanCodesDict[Node.value] = "0"
        else:
            HuffmanCodesDict[Node.value] = prevStr
    else:
        get_codes(prevStr+"0", Node.leftchild)
        get_codes(prevStr+"1", Node.rightchild)


# function to write the huffman codes for each pixel to file but as bits
# not as bytes of 0s and 1s. Builds a byte from the 0s and 1s and writes
# to file one byte at a time
def make_file(pixelList, width, height):
    f = open("output.bmp", "w")
    tempbyte = 0
    temppos = 0
    for i in range(height):
        for j in range(width):
            code = HuffmanCodesDict[pixelList[i][j]]
            for ch in code:
                if ch == '1':
                    temp = 1 << temppos
                    tempbyte = temp | tempbyte
                temppos +=1
                if temppos == 8:
                    temppos = 0
                    f.write(chr(tempbyte))
                    tempbyte = 0
    if temppos > 0:
        f.write(chr(tempbyte))
    print("NEW SIZE OF FILE IS: " ,f.tell())


# Was in hufman.py, used to generate a 2-d array of numbers
# to encode using huffman
def generate_array():
    arr = []
    height = 8
    width = 8
    for i in range(height):
        arr.append([])
        for j in range(width):
            if i % 2 == 0:
                arr[i].append(random.randrange(10, 20))
            else:
                arr[i].append((i + j) % 8)
    return arr, width, height


# was in hufman.py. Used to write hufman code for each
# pixel into a file as bits of 0s and 1s. It forms a giant string first
# and uses BitArray to convert to bits
def encode(huffman_codes, arr):
    encoded_str = ""
    for list in arr:
        for num in list:
            encoded_str += huffman_codes.get(num)
    return BitArray(bin=encoded_str).tobytes()

# main() to drive sunday huffman encoding
def main():
    input_array, width, height = generate_array()
    frequency_dict = get_frequencies(input_array)
    root = make_huffman_tree(frequency_dict)
    huffman_codes = {}
    get_huffman_codes(root, huffman_codes, "")
    encoded_str = encode(huffman_codes, input_array)
    print "ENCODED IS", encoded_str

    decoded_list = decode(encoded_str,root)
    decoded_arr = []
    for i in range(height):
        decoded_arr.append([0] * width)
        for j in range(width):
            decoded_arr[i][j] = decoded_list[i*width+j]

    print "input array:"
    print_2d_array(input_array)
    print "decoded array:"
    print_2d_array(decoded_arr)



# from heapq import *
# from heapq import heapify
#
# class Node(object):
#     leftchild = None
#     rightchild = None
#     value = None
#     freq = None
#
#     def __init__(self, value, weight):
#         self.value = value
#         self.freq = weight
#
#     def set_children(self, left, right):
#         self.leftchild = left
#         self.rightchild = right
#
#
#     def __repr__(self):
#         return "[{}-{}-{}-{}]".format(self.value, self.freq, self.leftchild, self.rightchild)
#
#     def __cmp__(self, other):
#         return cmp(self.freq, other.freq)
#
#
# CodesDict = {}
#
#

#
# def huffmanDecode (dictionary, huffmanCode):
#     res = ""
#     while huffmanCode:
#         for k in dictionary:
#             if huffmanCode.startswith(k):
#                 res += dictionary[k]
#                 huffmanCode = huffmanCode[len(k):]
#     return res


# main to test huffman program
# def main():
#     s = "aababcabcd"
#     print s
#     freqlist = {}
#     for num in s:
#         if num in freqlist:
#             freqlist[num] += 1
#         else:
#             freqlist[num] = 1
#     huffmanEncode(freqlist)
#     huffmanCodes = "".join([CodesDict[key] for key in s])
#     print "AFTER ENCODING ", huffmanCodes
#     invCodesDict = {v:k for k,v in CodesDict.iteritems()}
#     originalstr = huffmanDecode(invCodesDict,huffmanCodes)
#     print "AFTER DECODING ", originalstr
#     print CodesDict
# #    p = np.array(pixelList)
#     img = Image.fromarray(p)
#     img.show()
#
# main()




# TRYING THE OPTIMAL ENCODE AND DECODES
#
# def decode_2(encoded_str, root):
#     l = []
#     node = root
#     for charac in encoded_str:
#         for i in range(0, 8):
#             if ord(charac) & 1 << i != 0:
#                 node = node.right_child
#             else:
#                 node = node.left_child
#             if node.node_value != None:
#                 l.append(node.node_value)
#                 node = root
#     return l
#
# def encode_2(huffman_codes, arr):
#     f = open("output.bmp", "w")
#     tempbyte = 0
#     temppos = 0
#     en_str = 0
#     for list in arr:
#         for num in list:
#             code = huffman_codes[num]
#             for ch in code:
#                 if ch == '1':
#                     temp = 1 << temppos
#                     tempbyte = temp | tempbyte
#                 temppos += 1
#                 if temppos == 8:
#                     temppos = 0
#                     en_str += tempbyte
#                     # f.write(chr(tempbyte))
#                     tempbyte = 0
#     if temppos > 0:
#         en_str += tempbyte
#         # f.write(chr(tempbyte))
#     return en_str
