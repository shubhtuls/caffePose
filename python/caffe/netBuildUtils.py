from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from caffe import netSpec as NS

# helper function for common structures
def addConvRelu(net, suffix, bottom, ks, nout, stride=1, pad=0, group=1, weight_blobs=None):
    if weight_blobs is None:
      net.addLayer('conv'+suffix,'Convolution', bottom, 'conv'+suffix, 
                                 kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    else:
      net.addLayer('conv'+suffix, 'Convolution', bottom, 'conv'+suffix,
                                kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group, param=weight_blobs)

    net.addLayer('relu'+suffix, 'ReLU', 'conv'+suffix,[] , in_place=True)
    return 'conv'+suffix

def get_weight_blobs(lr_mult = [1,2], decay_mult=[1,0], name=None):
	if name is None:
		return [{'lr_mult':lr_mult[i],'decay_mult':decay_mult[i]} for i in range(len(lr_mult))]
	else:
		return [{'lr_mult':lr_mult[i],'decay_mult':decay_mult[i], 'name':name[i]} for i in range(len(lr_mult))]
    

def addFcRelu(net, suffix, bottom, nout, **kwargs):
	weight_blobs = kwargs.pop('weight_blobs', None)
	print(kwargs)
	if weight_blobs is None:
		net.addLayer('fc'+suffix, 'InnerProduct',bottom,'fc'+suffix,
			num_output=nout, **kwargs)
	else:
		net.addLayer('fc'+suffix, 'InnerProduct', bottom, 'fc'+suffix,num_output=nout, param=weight_blobs, **kwargs)
	net.addLayer('relu'+suffix, 'ReLU', 'fc'+suffix, [], in_place=True)
	return 'fc'+suffix

def addMaxPool(net, suffix, bottom, ks, stride=1):
	net.addLayer('pool'+suffix, 'Pooling', bottom,'pool'+suffix, pool=NS.P.Pooling.MAX, kernel_size=ks, stride=stride)
	return 'pool'+suffix

def addAlexNetTillConv(net, learn, img_blob_name = 'data', suffix=''):
	conv_lr_mult = [1.*learn, 2.*learn]; conv_decay_mult= [1.*learn, 0.*learn];
	
	print(suffix)    
	
	top_name = addConvRelu(net, '1'+suffix,img_blob_name, 11, 96, stride=4, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '1'+suffix, top_name, 3, stride=2)
	net.addLayer('lrn1', 'LRN', top_name, 'norm1',local_size=5, alpha=1e-4, beta=0.75)
	
	top_name = addConvRelu(net,'2'+suffix, 'norm1', 5, 256, pad=2, group=2, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '2'+suffix, top_name, 3, stride=2)
	net.addLayer('lrn2','LRN',top_name, 'norm2', local_size=5, alpha=1e-4, beta=0.75)
	
	top_name = addConvRelu(net, '3'+suffix , 'norm2', 3, 384, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	
	top_name = addConvRelu(net, '4'+suffix ,top_name, 3, 384, pad=1, group=2,weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	
	top_name = addConvRelu(net, '5'+suffix ,top_name, 3, 256, pad=1, group=2,weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '5'+suffix, top_name, 3, stride=2)
	return top_name

def addAlexnetFcLayers(net, learn, inputBlobName = 'pool5', suffix=''):
	fc_lr_mult = [1.*learn, 2.*learn]; fc_decay_mult= [1.*learn, 0.*learn];
	top_name = addFcRelu(net, '6'+suffix, inputBlobName, 4096, weight_blobs=get_weight_blobs(lr_mult = fc_lr_mult, decay_mult = fc_decay_mult))
	net.addLayer('drop6'+suffix, 'Dropout',top_name, [], in_place=True)
	top_name = addFcRelu(net, '7'+suffix, top_name, 4096, weight_blobs=get_weight_blobs(lr_mult = fc_lr_mult, decay_mult = fc_decay_mult))
	net.addLayer('drop7'+suffix, 'Dropout', top_name, [], in_place=True)
	return top_name    

def addVggTillConv(net, learn, img_blob_name = 'data', suffix=''):
	conv_lr_mult = [1.*learn, 2.*learn]; conv_decay_mult= [1.*learn, 0.*learn];
	print(suffix)
	
	top_name = addConvRelu(net, '1_1'+suffix,img_blob_name, 3, 64, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '1_2'+suffix,top_name, 3, 64, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '1'+suffix, top_name, 2, stride=2)
	
	top_name = addConvRelu(net, '2_1'+suffix,top_name, 3, 128, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '2_2'+suffix,top_name, 3, 128, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '2'+suffix, top_name, 2, stride=2)
	
	top_name = addConvRelu(net, '3_1'+suffix,top_name, 3, 256, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '3_2'+suffix,top_name, 3, 256, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '3_3'+suffix,top_name, 3, 256, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '3'+suffix, top_name, 2, stride=2)
	
	top_name = addConvRelu(net, '4_1'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '4_2'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '4_3'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '4'+suffix, top_name, 2, stride=2)
	
	top_name = addConvRelu(net, '5_1'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '5_2'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addConvRelu(net, '5_3'+suffix,top_name, 3, 512, stride=1, pad=1, weight_blobs=get_weight_blobs(lr_mult = conv_lr_mult, decay_mult = conv_decay_mult))
	top_name = addMaxPool(net, '5'+suffix, top_name, 2, stride=2)
	return top_name