#!/usr/bin/env bash
# Create the VOC2007 lmdb inputs

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/".."
source $SCRIPT_DIR/load_settings.bash

RESIZE_WIDTH=256
RESIZE_HEIGHT=256

#if [-d $DATA_RESIZED_DIR]
#  echo "--- INFO - Images not yet resized. Resizing..."
#  echo "--- Finished resizing images"
#fi

echo "--- INFO - Preparing image filename lists..."
$UTILITY_SCRIPT_DIR/prepare_image_list.py $DATASET_DIR/train.txt $DATABASE_DIR/train.txt
$UTILITY_SCRIPT_DIR/prepare_image_list.py $DATASET_DIR/val.txt $DATABASE_DIR/test.txt    # We don't have the actual test set, so use the validation set for testing.

echo "--- INFO - Creating train lmdb..."

# Remove old databases
rm -r $DATABASE_DIR/*_lmdb

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/train.txt \
    $DATABASE_DIR/train_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/train.txt $DATABASE_DIR/train_labels_lmdb

echo "--- INFO - Creating test lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/test.txt \
    $DATABASE_DIR/test_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/test.txt $DATABASE_DIR/test_labels_lmdb

echo "--- INFO - Creating 1 image debug train lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/train_debug_1.txt \
    $DATABASE_DIR/train_debug_1_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/train_debug_1.txt $DATABASE_DIR/train_labels_debug_1_lmdb

echo "--- INFO - Creating 1 image debug test lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/test_debug_1.txt \
    $DATABASE_DIR/test_debug_1_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/test_debug_1.txt $DATABASE_DIR/test_labels_debug_1_lmdb

echo "--- INFO - Creating 10 image debug train lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/train_debug_10.txt \
    $DATABASE_DIR/train_debug_10_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/train_debug_10.txt $DATABASE_DIR/train_labels_debug_10_lmdb

echo "--- INFO - Creating 10 image debug test lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS_DIR/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=True \
    $IMAGE_DIR \
    $DATABASE_DIR/test_debug_10.txt \
    $DATABASE_DIR/test_debug_10_lmdb

$UTILITY_SCRIPT_DIR/prepare_image_labels.py $DATABASE_DIR/train_debug_10.txt $DATABASE_DIR/teset_labels_debug_10_lmdb

echo "--- INFO - Done"
