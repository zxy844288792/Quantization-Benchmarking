# zoo_runtime.py
import os
from metadata import zoo_metadata 
import numpy as np
import glob

def get_val_data(rec_val,
                 batch_size,
                 num_workers=4,
                 shuffle=False):
    from mxnet import gluon
    import mxnet as mx
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = shuffle,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn

class ZooRuntime:
    def __init__(self, model_name, graph, lib, params):
        self.model_name = model_name
        self.graph = graph
        self.lib = lib
        self.params = params

    def run_tvm(self):
        self._model.run()
        return self._model.get_output(0)

    def load_tvm_model(self, ctx):
        from tvm.contrib import graph_runtime
        model = graph_runtime.create(self.graph, self.lib, ctx)
        model.set_input(**self.params)
        return model

    def load_input(self, x, input_name):
        import tvm
        self._model.set_input(input_name,tvm.nd.array(x))

    def performance(self, num_warm_up, num_iteration):
        import time
        data = np.random.random(self.input_shape)
        data = data.astype('float32')
        self.load_input(data, self.input_name)
        # warm up
        for i in range(num_warm_up):
            self._model.run()

        # performance benchmark
        total_time = 0
        for i in range(num_iteration):
            start_time = time.time()
            self._model.run()
            end_time = time.time()
            total_time = total_time + end_time - start_time
        
        return total_time / num_iteration

class GluonImageClassifierRuntime(ZooRuntime):
    def __init__(self, model_name, graph, lib, params):
        from mxnet.gluon.model_zoo.vision import get_model
        self.zoo = 'gluon_image_classifier'
        super(GluonImageClassifierRuntime, self).__init__(model_name, graph, lib, params)
        # TODO add shape logic
        self.height = 224
        self.width = 224
        self.classes = 1000
        self.channels = 3
        self.batch_size = 1
        self.input_shape = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
        )
        # TODO ctx logic
        self.input_name = 'data'
        import tvm
        self._model = self.load_tvm_model([tvm.cpu(0)])

    def evaluate(self):
        import mxnet as mx
        val_data, batch_fn = get_val_data('~/.mxnet/datasets/imagenet/rec/val.rec', 1)
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            # TODO ctx
            data, label = batch_fn(batch, [mx.cpu(0)])
            self.load_input(data[0].asnumpy(), self.input_name)
            out_arr = self.run_tvm()
            acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
            acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])
            
            if not (i + 1) % 1000:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                nsamples = (i + 1)
                print('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
        
        return top1, top5

class TFImageClassifierRuntime(ZooRuntime):
    def __init__(self, model_name, graph, lib, params):
        super(TFImageClassifierRuntime, self).__init__(model_name, graph, lib, params)
        self.zoo = 'tf_image_classifier'
        # TODO add shape logic
        self.height = 224
        self.width = 224
        self.classes = 1000
        self.channels = 3
        self.batch_size = 1
        self.input_shape = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
        )
        self.input_name = zoo_metadata[self.zoo][model_name][2]
        import tvm
        self._model = self.load_tvm_model([tvm.cpu(0)])

    def evaluate(self):
        import cv2
        image_path = '/home/ubuntu/.mxnet/datasets/imagenet/val/'
        all_class_path = sorted(glob.glob(image_path+'*'))
        total = 0
        top1_score = 0
        top5_score = 0
        label = 0
        for cur_class in all_class_path:
            all_image = glob.glob(cur_class+'/*')
            for image in all_image:
                total = total + 1
                im = cv2.imread(image)
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, dsize=(224, 224))
                im = np.expand_dims(im, axis=0)
                im = im.astype('float32')
                self.load_input(im, self.input_name)
                out_arr = self.run_tvm().asnumpy()

                if np.argmax(out_arr) == label:
                    top1_score = top1_score + 1
                if label in np.argsort(out_arr)[0][-5:]:
                    top5_score = top5_score + 1
                if not total % 1000:
                    print('[%d samples] validation: acc-top1=%f acc-top5=%f', total, top1_score/total, top5_score/total)
            label = label + 1

        return top1_score/total, top5_score/total

zoo_runtimes = {
    'gluon_image_classifier': GluonImageClassifierRuntime,
    'tf_image_classifier': TFImageClassifierRuntime,
}
