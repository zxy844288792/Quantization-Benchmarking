# zoo_compile.py
import os
import sys
import numpy as np
from metadata import model_zoo_models, zoo_metadata, TARGET_MAP

def compile_model_tvm(mod, params, target, quantized):
    from tvm import relay
    if quantized:
        graph, lib, params = relay.build(mod, target=target)
    else:
        graph, lib, params = relay.build(mod, params=params, target=target)
    
    return graph, lib, params

class ZooCompiler:
    def __init__(self, model_name, target, local_path, quantized):
        self.model_name = model_name
        self.target = target
        self.local_path = local_path
        self.target = TARGET_MAP[target]
        self.quantized = quantized

    def compile(self):
        return compile_model_tvm(self.mod, self.params, self.target, self.quantized)
    
    def load(self):
        pass

class GluonImageClassifierCompiler(ZooCompiler):
    def __init__(self, model_name, target, local_path, quantized):
        super(GluonImageClassifierCompiler, self).__init__(model_name,
                                                              target,
                                                              local_path,
                                                              quantized)
        self.zoo = 'gluon_image_classifier'
        self.channels = 3
        # shape logic need to be added 
        self.height = 224
        self.width = 224
        self.input_shape = [
            1,
            self.channels,
            self.height,
            self.width,
        ]
        self.model_key = model_zoo_models[self.zoo][model_name]
        self.input_shape = {"data": self.input_shape}
        self.mod, self.params = self.load_model()
        from tvm import relay
        if self.quantized:
            with relay.quantize.qconfig(store_lowbit_output=False):
                func = self.mod['main']
                self.mod = relay.quantize.quantize(func,params=self.params)
 
    def load_model(self):
        from mxnet.gluon.model_zoo import vision
        model = vision.get_model(self.model_key,pretrained=True)
        from tvm import relay
        return relay.frontend.from_mxnet(model, shape=self.input_shape)


class TFImageClassifierCompiler(ZooCompiler):
    def __init__(self, model_name, target, local_path, quantized):
        super(TFImageClassifierCompiler, self).__init__(model_name,
                                                              target,
                                                              local_path,
                                                              quantized)
        self.zoo = 'tf_image_classifier'
        self.channels = 3
        self.height, self.width, self.input_name = zoo_metadata[self.zoo][model_name]
        self.input_shape = [
            1,
            self.height,
            self.width,
            self.channels,
        ]
        self.model_key = model_zoo_models[self.zoo][model_name]
        self.input_shape = {self.input_name: self.input_shape}
        # load model
        self.mod, self.params = self.load_model()
        from tvm import relay
        if self.quantized:
            with relay.quantize.qconfig(store_lowbit_output=False):
                func = self.mod['main']
                self.mod = relay.quantize.quantize(func,params=self.params)

    def load_model(self):
        # TODO path
        from tvm.relay.frontend.tensorflow_parser import TFParser
        model_path = '/home/ubuntu/tf/'+self.model_key+'_frozen.pb'
        #model_path = '/home/ubuntu/tf/'+self.model_path
        tf_graph = TFParser(model_path).parse()
        from tvm import relay
        return relay.frontend.from_tensorflow(tf_graph, layout='NCHW', shape=self.input_shape)

zoo_compilers = {
    'gluon_image_classifier': GluonImageClassifierCompiler,
    'tf_image_classifier': TFImageClassifierCompiler,
}
