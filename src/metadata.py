# metadata.py
from collections import defaultdict

model_zoos = [
    'gluon_image_classifier',
    'tf_image_classifier',
]

model_zoo_models = defaultdict(dict)

model_zoo_models['gluon_image_classifier']['resnet18_v1'] = 'resnet18_v1'
model_zoo_models['gluon_image_classifier']['resnet34_v1'] = 'resnet34_v1'
model_zoo_models['gluon_image_classifier']['mobilenet1.0'] = 'mobilenet1.0'

model_zoo_models['tf_image_classifier']['resnet50_v1'] = 'resnet_v1_50'

zoo_metadata = defaultdict(dict)

zoo_metadata['tf_image_classifier'] = {
        'resnet50_v1': (224, 224, 'input'),
}

TARGET_MAP = {
    'c5'    : 'llvm -mcpu=skylake-avx512',
    'm5'    : 'llvm -mcpu=skylake-avx512',
    'c4'    : 'llvm -mcpu=core-avx2',
    'm4'    : 'llvm -mcpu=core-avx2',
}
