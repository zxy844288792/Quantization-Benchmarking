# metadata.py
from collections import defaultdict

model_zoos = [
    'gluon_image_classifier',
    'tf_image_classifier',
]

model_zoo_models = defaultdict(dict)

model_zoo_models['gluon_image_classifier'] = {
    'resnet18_v1': 'resnet18_v1', 
    'resnet34_v1': 'resnet34_v1',
    'resnet50_v1': 'resnet50_v1',
    'resnet101_v1': 'resnet101_v1',
    'mobilenet1.0': 'mobilenet1.0',
    'vgg11': 'vgg11',
    'vgg16': 'vgg16',
}

model_zoo_models['tf_image_classifier'] = {
    'resnet50_v1': 'resnet_v1_50',
    'resnet101_v1': 'resnet_v1_101',
    'resnet152_v1': 'resnet_v1_152',
    'vgg16': 'vgg_16',
    #'mobilenet1.0': 'mobilenet_v1_1.0_224',
}


zoo_metadata = defaultdict(dict)

zoo_metadata['tf_image_classifier'] = {
        'resnet50_v1': (224, 224, 'input'),
        'resnet101_v1': (224, 224, 'input'),
        'resnet152_v1': (224, 224, 'input'),
        'vgg16': (224, 224, 'input'),
        'mobilenet1.0': (224, 224, 'input'),
}

TARGET_MAP = {
    'c5'    : 'llvm -mcpu=skylake-avx512',
    'm5'    : 'llvm -mcpu=skylake-avx512',
    'c4'    : 'llvm -mcpu=core-avx2',
    'm4'    : 'llvm -mcpu=core-avx2',
}
