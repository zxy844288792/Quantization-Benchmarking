from argparse import ArgumentParser
import threading
import logging
import datetime
import time
from metadata import model_zoos, model_zoo_models
import zoo_compile
import zoo_runtime

def worker(b):
    # target logic need to be added
    target = 'c5'
    for model_zoo_name in model_zoos:
        logging.info("start model zoo: %s" % (model_zoo_name))
        for model_name in model_zoo_models[model_zoo_name]:
            logging.info("start model: %s" % (model_name))
            compiler = zoo_compile.zoo_compilers[model_zoo_name](model_name, target, '', False)
            try:
                quantized_compiler = zoo_compile.zoo_compilers[model_zoo_name](model_name, target, '', True)
                logging.info('quantization succeed for %s %s' % (model_zoo_name, model_name))
            except:
                logging.info('quantization failed for %s %s' % (model_zoo_name, model_name))
            # original model compilation
            graph, lib, params = compiler.compile()

            # original model performance
            runtime = zoo_runtime.zoo_runtimes[model_zoo_name](model_name, graph, lib, params)
            latency = runtime.performance(20, 100)
            logging.info('original %s %s latency: %f' % (model_zoo_name, model_name, latency))

            # original model evaluation
            acc1, acc5 = runtime.evaluate() 
            logging.info('original %s %s top1: %f top5: %f' % (model_zoo_name, model_name, acc1, acc5))

            # quantized model compilation
            graph, lib, params = quantized_compiler.compile()

            # quantized model performance
            runtime = zoo_runtime.zoo_runtimes[model_zoo_name](model_name, graph, lib, params)
            latency = runtime.performance(20, 100)
            logging.info('quantized %s %s latency: %f' % (model_zoo_name, model_name, latency))

            # quantized mode levaluation
            acc1, acc5 = runtime.evaluate()
            logging.info('quantized %s %s top1: %f top5: %f' % (model_zoo_name, model_name, acc1, acc5))
    b.wait()
    print('done')

def main():
    # since we are not targeting on multiple insance type here, the argument is not useful now
    parser = ArgumentParser(description='launching quantization benchmark')
    parser.add_argument('--region_name', type=str, help='name of region. default: us-west-1',
                        default='us-west-2')

    logging.basicConfig(filename='%s.log' % datetime.date.today().strftime("%B_%d_%Y"), level=logging.INFO)
    
    args = parser.parse_args()

    b = threading.Barrier(2)
    thread = threading.Thread(target=worker, args=(b,))
    thread.start()

    logging.info('waiting for all test threads to complete...')
    b.wait()
    thread.join()

    logging.info("All Test Done")

if __name__ == '__main__':
    main()
