import argparse
import sys
import os
import shutil
import zipfile
import time

# torchlight
import torchlight
from torchlight import import_class

from processor.processor import init_seed
init_seed(0)

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()

    processors['pretrain_actclr'] = import_class('processor.pretrain_actclr.ActCLR_Processor')

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    parser.add_argument("--local_rank", default=0, type=int)
    
    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--config":
            sys_argv = sys.argv[i:]
            break

    p = Processor(sys_argv)

    p.init_environment()
    
    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)
            
    print(int(os.environ["LOCAL_RANK"]))

    p.start(int(os.environ["LOCAL_RANK"]))