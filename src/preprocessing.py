import utils
import numpy as np

def preprocessing(img, pre_category, threshold=0.875):
    fn = preprocessing_fn_map[pre_category]
    return fn(img,threshold=threshold)

def vgg_preprocessing(img, threshold=0.875):
    import cv2
    mean_rgb = [123.68, 116.779, 103.939]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = img - np.array(mean_rgb)
    img = img.astype('float32')
    return img

def inception_preprocessing(img, threshold=0.875):
    import cv2
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img =  utils.crop_center(img, threshold=threshold)
    img = cv2.resize(img, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 256
    img = (img - 0.5) * 2
    img = img.astype('float32')
    return img


preprocessing_fn_map = {
    'vgg_preprocessing': vgg_preprocessing,
    'inception_preprocessing': inception_preprocessing,
}
