import os
from bitstring import BitArray


class Node(object):
    left_child = None
    right_child = None
    node_freq = None
    node_value = None

    def __init__(self, left, right, value, freq):
        self.left_child = left
        self.right_child = right
        self.node_value = value
        self.node_freq = freq

    def __cmp__(self, other_node):
        return cmp(self.node_freq, other_node.node_freq)


def encode_driver(pixel_array, width, height):
    frequency_dict = get_frequencies(pixel_array)
    root = make_huffman_tree(frequency_dict)
    huffman_codes = {}
    get_huffman_codes(root, huffman_codes, "")
    encode_bits_to_file(pixel_array, huffman_codes, width, height)
    print "NEW SIZE OF FILE IS: ", os.stat("encodeoutput.bmp").st_size
    print "ENCODING IS DONE!!!"
    return root


# This following functions main, decoder() and write_to_file() are used to
# test the hufman implementation by itself.
def main():
    words = "the quick brown fox jumps over the lazy dog"
    frequency_dict = get_frequencies(words)
    root = make_huffman_tree(frequency_dict)
    huffman_codes = {}
    get_huffman_codes(root, huffman_codes, "")
    write_to_file(words, huffman_codes)
    print "NEW SIZE OF FILE IS: ", os.stat(".bmp").st_size
    print "ENCODING IS DONE!!!"
    decoder(root)



def decoder(root):
    with open("output.bmp", "r") as enc_file:
        encoded_str = enc_file.read().replace("\n", "")
        decode_driver(encoded_str, root, 1, 1)

def write_to_file(words, huffman_codes_dict):
    with open("output.bmp", "w") as out_file:
        tempbyte = 0
        temppos = 0
        for char in words:
                code = huffman_codes_dict[char]
                for ch in code:
                    if ch == '1':
                        temp = 1 << temppos
                        tempbyte = temp | tempbyte
                    temppos += 1
                    if temppos == 8:
                        temppos = 0
                        out_file.write(chr(tempbyte))
                        tempbyte = 0
        if temppos > 0:
            out_file.write(chr(tempbyte))


def get_frequencies(input_array):
    freqs = {}
    for list in input_array:
        for num in list:
            key = num
            if key in freqs:
                freqs[key] += 1
            else:
                freqs[key] = 1
    return freqs


def make_huffman_tree(frequency_list):
    nodes_list = []
    for key, value in frequency_list.iteritems():
        nodes_list.append(Node(None, None, key, value))
    nodes_list.sort()
    while len(nodes_list) > 1:
        left = nodes_list.pop(0)
        right = nodes_list.pop(0)
        new_node = Node(left, right, None, (left.node_freq + right.node_freq))
        nodes_list.append(new_node)
        nodes_list.sort()

    return nodes_list[0]


def get_huffman_codes(node, dict, curr_bits):
    if node.node_value is None:
        get_huffman_codes(node.left_child, dict, curr_bits + "0")
        get_huffman_codes(node.right_child, dict, curr_bits + "1")
    else:
        if curr_bits is None:
            dict[node.node_value] = "0"
        else:
            dict[node.node_value] = curr_bits


def encode_bits_to_file(pixel_list, huffman_codes_dict, width, height):
    with open("encodeoutput.bmp", "w") as out_file:
        tempbyte = 0
        temppos = 0
        for i in range(height):
            for j in range(width):
                code = huffman_codes_dict[pixel_list[i][j]]
                for ch in code:
                    if ch == '1':
                        temp = 1 << temppos
                        tempbyte = temp | tempbyte
                    temppos += 1
                    if temppos == 8:
                        temppos = 0
                        out_file.write(chr(tempbyte))
                        tempbyte = 0
        if temppos > 0:
            out_file.write(chr(tempbyte))


def decode_driver(encoded_str, root, height, width):
    decoded_list = decode_2(encoded_str, root)
    print "length of list after decode is", len(decoded_list)
    decoded_arr = []
    a = 0
    for i in range(height):
        decoded_arr.append([])
        for j in range(width):
            # decoded_arr[i].append(a)
            # a = a + 1
            decoded_arr[i].append(float(decoded_list[(i * width) + j]))

    return decoded_arr


def decode_2(encoded_str, root):
    l = []
    node = root
    for charac in encoded_str:
        for i in range(0, 8):
            if (ord(charac) & 1 << i) != 0:
                node = node.right_child
            else:
                node = node.left_child
            if node.node_value != None:
                l.append(node.node_value)
                node = root
    return l


def decode(encoded_str, root):
    l = []
    bit_array = BitArray(bytes=encoded_str).bin
    print "SIZE OF BIT_ARRAY IS ", len(bit_array)
    node = root
    for ch in bit_array:
        if ch == "1":
            node = node.right_child
        else:
            node = node.left_child
        if node.node_value != None:
            l.append(node.node_value)

            node = root
    return l


def print_2d_array(array):
    for row in array:
        print row

