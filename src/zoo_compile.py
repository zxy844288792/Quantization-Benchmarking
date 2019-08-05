# zoo_compile.py
import os
import sys
import numpy as np
from metadata import model_zoo_models, zoo_metadata, TARGET_MAP

def compile_model_tvm(mod, params, target, quantized):
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

class GluonImageNetClassifierCompiler(ZooCompiler):
    def __init__(self, model_name, target, local_path):
        super(GluonImageNetClassifierCompiler, self).__init__(model_name,
                                                              target,
                                                              local_path)
        self.zoo = 'gluon_imagenet_classifier'
        self.channels = 3
        self.height = 299 if utils.is_inception(model_name) else 224
        self.width = 299 if utils.is_inception(model_name) else 224
        self.input_shape = [
            1,
            self.channels,
            self.height,
            self.width,
        ]
        self.model_key = model_zoo_models[self.zoo][model_name]
        self.input_shape = "{\"data\": %s}" % self.input_shape
        from tvm import relay
        self.mod, self.params = relay.frontend.from_mxnet(self.model_key, shape=self.input_shape)
        if self.quantized:
            func = self.mod['main']
            self.mod = relay.quantize.quantize(func,params=self.params)
    
    def compile(self, s3_bucket, backend='TVM'):
        return compile_model_tvm(self.mod, self.params, self.target, self.quantized)


zoo_compilers = {
    'gluon_imagenet_classifier': GluonImageNetClassifierCompiler,
    #'tf_imagenet_classifier': TFImageNetClassifierCompiler,
}
