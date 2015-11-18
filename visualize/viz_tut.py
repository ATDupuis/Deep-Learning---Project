import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
out = net.forward()
print("Predicted class is #{}.".format(out['prob'][0].argmax()))

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
plt.savefig('viz_results/demo/input.png')

# CPU mode
net.forward()  # call once for allocation

[(k, v.data.shape) for k, v in net.blobs.items()]

[(k, v[0].data.shape) for k, v in net.params.items()]



# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


    # the parameters are a list of [weights, biases]
#conv layer 1 filters
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.savefig('viz_results/demo/conv1_filter.png')

#conv layer 1 output
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)
plt.savefig('viz_results/demo/conv1_output.png')

# conv layer 2 filters
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))
plt.savefig('viz_results/demo/conv2_filter.png')

# conv layer 2 output
feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)
plt.savefig('viz_results/demo/conv2_output.png')

#conv layer 3 output
feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
plt.savefig('viz_results/demo/conv3_output.png')


#conv layer 4 output
feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
plt.savefig('viz_results/demo/conv4_output.png')

#conv layer 5 output
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
plt.savefig('viz_results/demo/conv5_output.png')

#pool layer 5 output
feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
plt.savefig('viz_results/demo/pool5_output.png')

#first fully connnected layer -- output and histogram of positive values
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig('viz_results/demo/fc6_output_hist.png')


#second fully connected layer -- output and histogram of positive values
plt.clf() #clear image
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig('viz_results/demo/fc7_output_hist.png')

#final output probability
plt.clf() #clear image
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.savefig('viz_results/demo/final_output.png')
