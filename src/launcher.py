from argparse import ArgumentParser
import threading
import logging
import datetime
import time
import benchmark

def worker(b):
    
    b.wait()
    print('done', sleeptime)

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
