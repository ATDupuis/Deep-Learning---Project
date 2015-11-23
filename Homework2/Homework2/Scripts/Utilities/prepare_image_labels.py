#!/usr/bin/env python
# This script adds creates an lmdb databases containing the labels for each image.

import caffe
import lmdb
import numpy
import os
import sys

if __name__ == "__main__":
    image_list_filename = sys.argv[1]
    database_name = sys.argv[2]

    print "Source image list filename: ", image_list_filename


    # Obtain all of the image filenames
    print "Obtaining image filenames..."
    with open(image_list_filename, 'r') as image_list_file:
        image_filenames = [os.path.splitext(line.split()[0])[0] for line in image_list_file.read().splitlines()]

    # Obtain the each image's labels
    print "Obtaining image labels..."
    categories = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    category_indices = dict(zip(categories, range(len(categories))))

    image_labels = dict(zip(image_filenames, [[0 for col in range(len(categories))] for image_index in range(len(image_filenames))]))

    for category in categories:
        category_index = category_indices[category]
        category_list_filename = os.path.join("Data", "VOC2007", "ImageSets", "Main", category + "_trainval.txt")
        with open(category_list_filename) as category_list_file:
            lines_splits = [line.split() for line in category_list_file.readlines()]
            category_image_filenames = [line_splits[0] for line_splits in lines_splits if line_splits[1] == "1"]

            for category_image_filename in category_image_filenames:
                try:
                    image_labels[category_image_filename][category_index] = 1
                except:
                    # Ignore key errors. That just means this particular file was not in the provided dataset.
                    #print "Ignoring file: " + category_image_filename
                    pass



    print "Writing image labels to database: " + database_name
    database = lmdb.open(database_name, map_size=1000000)

    with database.begin(write=True) as transaction:
        for image_index in range(len(image_labels)):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = len(categories)
            datum.width = 1
            datum.data = numpy.uint8(numpy.array(image_labels[image_filenames[image_index]])).tostring()
            datum.label = 0  # Use a dummy label here. The actual labels are in the data field.
            id = '{:08}'.format(image_index)

            # The encode is only essential in Python 3
            transaction.put(id.encode('ascii'), datum.SerializeToString())

    print "Done"


    #print "Output image list filename: ", modified_image_list_filename

    #extension = ".jpg"
    #dummy_label = "0"

    #with open(image_list_filename, 'r') as image_list_file:
    #    lines = [''.join([line.strip(), extension, ' ', dummy_label, '\n']) for line in image_list_file.readlines()]

    #with open(modified_image_list_filename, 'w') as modified_image_list_file:
    #    modified_image_list_file.writelines(lines)
