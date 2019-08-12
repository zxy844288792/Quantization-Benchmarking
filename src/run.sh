#!/bin/bash
cd ~/tvm
git pull
cd build
make -j4
cd ..
cd python; python3 setup.py install --user; cd ..
cd topi/python; python3 setup.py install --user; cd ../..
cd nnvm/python; python3 setup.py install --user; cd ../..
cd ../Quantization-Benchmarking/src
python3 launcher.py --config_file config.txt
