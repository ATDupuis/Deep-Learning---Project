import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/mnist/mnist_autoencoder.prototxt',
                caffe_root + 'examples/mnist/mnist_autoencoder_iter_17990.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

# set net to batch size of 50
net.blobs['data'].reshape(50,1,28,28)

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/mnist_1.png',color=False))

#TODO: figure out the deprocess error thing
plt.imshow(np.squeeze(net.blobs['data'].data[0]))
plt.savefig('visualize/viz_results/ae/input.png')

# CPU mode
net.forward()  # call once for allocation

# check the output data dimension for all layers
[(k, v.data.shape) for k, v in net.blobs.items()]
# ('data', (100, 1, 28, 28))
# ('data_data_0_split_0', (100, 1, 28, 28))
# ('data_data_0_split_1', (100, 1, 28, 28))
# ('flatdata', (100, 784))
# ('flatdata_flatdata_0_split_0', (100, 784))
# ('flatdata_flatdata_0_split_1', (100, 784))
# ('encode1', (100, 1000))
# ('encode1neuron', (100, 1000))
# ('encode2', (100, 500))
# ('encode2neuron', (100, 500))
# ('encode3', (100, 250))
# ('encode3neuron', (100, 250))
# ('encode4', (100, 30))
# ('decode4', (100, 250))
# ('decode4neuron', (100, 250))
# ('decode3', (100, 500))
# ('decode3neuron', (100, 500))
# ('decode2', (100, 1000))
# ('decode2neuron', (100, 1000))
# ('decode1', (100, 784))
# ('decode1_decode1_0_split_0', (100, 784))
# ('decode1_decode1_0_split_1', (100, 784))
# ('cross_entropy_loss', ())
# ('decode1neuron', (100, 784))
# ('l2_error', ())


#check the parameter dimensions for each layer -- weight/filter
[(k, v[0].data.shape) for k, v in net.params.items()]
# ('encode1', (1000, 784))
# ('encode2', (500, 1000))
# ('encode3', (250, 500))
# ('encode4', (30, 250))
# ('decode4', (250, 30))
# ('decode3', (500, 250))
# ('decode2', (1000, 500))
# ('decode1', (784, 1000))


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


# encode1 filters
filters = net.params['encode1'][0].data #1000x784
filters = np.reshape(filters,(-1,28,28)) #1000x28x28
vis_square(filters)
plt.savefig('visualize/viz_results/ae/encode1_filters.png')

#encode1 output
feat = net.blobs['encode1'].data[0]
feat = np.reshape(feat, (-1,10,10)) # 1000 = 10x10x10
vis_square(feat,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode1_output.png')


# encode2 filters
filters = net.params['encode2'][0].data #500x1000
tst = np.reshape(filters,(-1,100,100)) #50x100x100
vis_square(tst)
plt.savefig('visualize/viz_results/ae/encode2_filters_100.png')
tst = np.reshape(filters,(-1,25,25)) #800x25x25
vis_square(tst)
plt.savefig('visualize/viz_results/ae/encode2_filters_25.png')


#encode2 output
feat = net.blobs['encode2'].data[0] #500
feat = np.reshape(feat, (-1,10,10)) # 500 = 5x10x10
vis_square(feat,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode2_output.png')

#encode3 filters
filters = net.params['encode3'][0].data #250x500
tst = np.reshape(filters,(-1,10,10)) #1250x10x10
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode3_filters_10.png')
tst = np.reshape(filters,(-1,250,250)) #2x250x250
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode3_filters_250.png')

#encode3 output
feat = net.blobs['encode3'].data[0] #250
tst = np.reshape(feat,(-1,5,5)) #10x5x5
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode3_output.png')

#encode4 filters
filters = net.params['encode4'][0].data #30x250
tst = np.reshape(filters,(-1,50,50))
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode4_filters_50.png')
tst = np.reshape(filters,(-1,10,10))
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/encode4_filters_10.png')


#decode4 filters
filters = net.params['decode4'][0].data #250x30
tst = np.reshape(filters,(-1,50,50))
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode4_filters_50.png')
tst = np.reshape(filters,(-1,10,10))
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode4_filters_10.png')


#decode3 filters
filters = net.params['decode3'][0].data #500x250
tst = np.reshape(filters,(-1,10,10)) #1250x10x10
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode3_filters_10.png')
tst = np.reshape(filters,(-1,250,250)) #2x250x250
vis_square(tst,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode3_filters_250.png')

#decode3 output
feat = net.blobs['decode3'].data[0] #500
feat = np.reshape(feat, (-1,10,10)) # 500 = 5x10x10
vis_square(feat,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode3_output.png')

# decode2 filters
filters = net.params['decode2'][0].data #500x1000
tst = np.reshape(filters,(-1,100,100)) #50x100x100
vis_square(tst)
plt.savefig('visualize/viz_results/ae/decode2_filters_100.png')
tst = np.reshape(filters,(-1,25,25)) #800x25x25
vis_square(tst)
plt.savefig('visualize/viz_results/ae/decode2_filters_25.png')


#decode2 output
feat = net.blobs['decode2'].data[0] #500
feat = np.reshape(feat, (-1,10,10)) # 1000 = 10x10x10
vis_square(feat,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode2_output.png')

# decode1 filters
filters = net.params['decode1'][0].data #1000x784
filters = np.reshape(filters,(-1,28,28)) #1000x28x28
vis_square(filters)
plt.savefig('visualize/viz_results/ae/decode1_filters.png')

#decode1 output
feat = net.blobs['decode1'].data[0]
feat = np.reshape(feat, (-1,28,28)) # 1000 = 1x28x28
vis_square(feat,padval=0.5)
plt.savefig('visualize/viz_results/ae/decode1_output.png')


