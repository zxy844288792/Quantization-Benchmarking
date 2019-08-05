from argparse import ArgumentParser
import threading
import logging
import datetime
import time
from metadata import model_zoos, model_zoo_models
import zoo_compile

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
            except:
                logging.info('quantization failed for %s %s' % (model_zoo_name, model_name))
            # original model compilation
            graph, lib, params = compiler.compile()



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
