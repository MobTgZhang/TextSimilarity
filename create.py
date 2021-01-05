import argparse
import logging
import sys
import os
from utils.config import MODEL_DIR
logger = logging.getLogger()
def set_default_args(parser):
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--rawpath', type=str, default='rawdata',
                       help='The raw data path')
    files.add_argument('--newpath', type=str, default='newdata',
                       help='The generated data path')
    files.add_argument('--model-name', type=str, default=None,
                       help='The generated data path')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='The generated data path')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type=bool, default=False,
                           help='Save model + optimizer state after each epoch')
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # Set model name
    if not args.model_name:
        import uuid
        import time

        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    return args
from preprocess.process import ProcessFile
if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser()
    args = set_default_args(parser)
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    ProcessFile(args.rawpath,args.newpath)