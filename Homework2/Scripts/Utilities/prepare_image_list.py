#!/usr/bin/env python
# This script adds the .jpg extension to every filename in a VOC2007 file
# list (train.txt, val.txt, trainval.txt) and a dummy label so that Caffe's
# convert_imageset can process them.

import sys;

if __name__ == "__main__":
    image_list_filename = sys.argv[1]
    modified_image_list_filename = sys.argv[2]

    print "Source image list filename: ", image_list_filename
    print "Output image list filename: ", modified_image_list_filename

    extension = ".jpg"
    dummy_label = "0"

    with open(image_list_filename, 'r') as image_list_file:
        lines = [''.join([line.strip(), extension, ' ', dummy_label, '\n']) for line in image_list_file.readlines()]

    with open(modified_image_list_filename, 'w') as modified_image_list_file:
        modified_image_list_file.writelines(lines)
