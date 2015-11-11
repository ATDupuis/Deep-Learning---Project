#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=project/test_examples/mnist/mnist_autoencoder_solver.prototxt
