# log_parser.py
import boto3
from argparse import ArgumentParser
from collections import defaultdict
import utils
from metadata import model_zoos, model_zoo_models

def email_gen(logfile, region_name):
    # upload the log file to s3
    #s3 = boto3.Session(profile_name='trinity').resource('s3', region_name=region_name)
    #utils.upload_to_s3(s3, 'quantization-benchmark-data', logfile)

    dic = defaultdict(dict)
    for model_zoo_name in model_zoos:
        for model_name in model_zoo_models[model_zoo_name]:
            dic[model_zoo_name][model_name] = {}

    with open(logfile, 'r') as log_file:
        for line in log_file:
            line = line.rstrip()
            if 'INFO:root:' in line:
                line = line[len('INFO:root:'):]
                if 'quantization failed' in line:
                    # quantization failed
                    model_zoo_name = line.split()[3]
                    model_name = line.split()[4]
                    dic[model_zoo_name][model_name]['quantized performance'] = ''
                    dic[model_zoo_name][model_name]['original performance'] = ''
                    dic[model_zoo_name][model_name]['quantized accuracy'] = ''
                    dic[model_zoo_name][model_name]['original accuracy'] = ''
                if 'latency' in line:
                    model_zoo_name = line.split()[1]
                    model_name = line.split()[2]
                    data = line.split()[4]
                    model_type = line.split()[0]
                    dic[model_zoo_name][model_name][model_type+' performance'] = data
                if 'top1' in line:
                    model_type = line.split()[0]
                    model_zoo_name = line.split()[1]
                    model_name = line.split()[2]
                    top1 = line.split()[4]
                    top5 = line.split()[6]
                    dic[model_zoo_name][model_name][model_type+' accuracy'] = top1 + '/' + top5


    print(dic)


def main():
    parser = ArgumentParser(description='process the log file')
    parser.add_argument('--logfile', type=str, help='file name of log file to be parsed', required=True)
    parser.add_argument('--region_name', type=str, help='default region', default='us-east-1')

    args = parser.parse_args()
    email_gen(args.logfile, args.region_name)

if __name__ == '__main__':
    main()
