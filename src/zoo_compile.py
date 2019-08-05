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

    def compile(self, s3_bucket, backend='TVM'):
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
        print(self.zoo)
        print(model_name)
        self.model_key = model_zoo_models[self.zoo][model_name]
        from mxnet.gluon.model_zoo import vision
        model = vision.get_model(self.model_key,pretrained=True)
        self.input_shape = {"data": self.input_shape}
        from tvm import relay
        self.mod, self.params = relay.frontend.from_mxnet(model, shape=self.input_shape)
        if self.quantized:
            func = self.mod['main']
            self.mod = relay.quantize.quantize(func,params=self.params)
    
    def compile(self):
        return compile_model_tvm(self.mod, self.params, self.target, self.quantized)


zoo_compilers = {
    'gluon_image_classifier': GluonImageClassifierCompiler,
    #'tf_imagenet_classifier': TFImageNetClassifierCompiler,
}
