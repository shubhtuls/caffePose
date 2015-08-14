"""Python net specification.

This module provides a way to write nets directly in Python, using a natural,
functional style. See examples/python_nets/caffenet.py for an example.

Currently this works as a thin wrapper around the Python protobuf interface,
with layers and parameters automatically generated for the "layers" and
"params" pseudo-modules, which are actually objects using __getattr__ magic
to generate protobuf messages.

Note that when using to_proto or Top.to_proto, names of intermediate blobs will
be automatically generated. To explicitly specify blob names, use the NetSpec
class -- assign to its attributes directly to name layers, and call
NetSpec.to_proto to serialize all assigned layers.

This interface is expected to continue to evolve as Caffe gains new capabilities
for specifying nets. In particular, the automatically generated layer names
are not guaranteed to be forward-compatible.
"""

from collections import OrderedDict

from .proto import caffe_pb2
from google import protobuf


def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [s for s in dir(layer) if s.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))

def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly."""

    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in item.iteritems():
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in val.iteritems():
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)



def getLayer(layer_name, layer_type, bottom_names,top_names, **kwargs):
  layer = caffe_pb2.LayerParameter()
  params = kwargs
  
  
  layer.name = layer_name
  layer.type = layer_type
  
  if isinstance(bottom_names, str):
    bottom_names=[bottom_names]

  if isinstance(top_names, str):
    top_names = [top_names]
  layer.bottom.extend(bottom_names)
  in_place = False
  if 'in_place' in params:
    in_place =params['in_place']
    del  params['in_place']
  if in_place:
    layer.top.extend(layer.bottom)
  else:
    layer.top.extend(top_names)
  
  #params
  for k, v in params.iteritems():
    if k.endswith('param'):
      assign_proto(layer, k, v)
    else:
      try:
        assign_proto(getattr(layer,
                    _param_names[layer_type] + '_param'), k, v)
      except (AttributeError, KeyError):
        assign_proto(layer, k, v)
  return layer



class NetSpec(object):
    """A NetSpec contains a set of Tops (assigned directly as attributes).
    Calling NetSpec.to_proto generates a NetParameter containing all of the
    layers needed to produce all of the assigned Tops, using the assigned
    names."""

    def __init__(self):
      self.net = caffe_pb2.NetParameter()
      self._param_names = param_name_dict()
      self.autoname_count = 0
      self.bottoms = set([])
      self.tops = set([])

    def addLayer(self,layer_name, layer_type, bottom_names,top_names, **kwargs):
     
      layer = caffe_pb2.LayerParameter()
      params = kwargs
      
      
      layer.name = layer_name
      layer.type = layer_type
     
      if isinstance(bottom_names, str):
        bottom_names=[bottom_names]

      if isinstance(top_names, str):
        top_names = [top_names]
      layer.bottom.extend(bottom_names)
      in_place = False
      if 'in_place' in params:
        in_place =params['in_place']
        del  params['in_place']
      if in_place:
        layer.top.extend(layer.bottom)
      else:
        layer.top.extend(top_names)
      
      #params
      for k, v in params.iteritems():
        if k.endswith('param'):
          assign_proto(layer, k, v)
        else:
          try:
            assign_proto(getattr(layer,
                        self._param_names[layer_type] + '_param'), k, v)
          except (AttributeError, KeyError):
            assign_proto(layer, k, v)
      self.net.layer.extend([layer])
      #add to set of bottoms and tops
      self.bottoms = self.bottoms | set(bottom_names)
      self.tops = self.tops | set(top_names)

    def addPrecookedLayer(self, layer):
      self.net.layer.extend([layer])


    def addInputBlobs(self,input_names_to_dims):
      for k, v in input_names_to_dims.items(): 
        print k
        self.net.input.extend([k])
        self.bottoms = self.bottoms | set([k])
        assign_proto(self.net, 'input_shape', [{'dim':v}])

    def getOutputBlobNames(self):
      return self.tops-self.bottoms

    def appendNet(self, secondnet):
      #extend inputs
      self.net.input.extend(secondnet.toProto().input)
      self.net.input_shape.extend(secondnet.toProto().input_shape)
      #extend layers
      self.net.layer.extend(secondnet.toProto().layer)


    def toProto(self):
      return self.net


class Parameters(object):
    """A Parameters object is a pseudo-module which generates constants used
    in layer parameters; e.g., Parameters().Pooling.MAX is the value used
    to specify max pooling."""

    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()



_param_names = param_name_dict()
P = Parameters()
