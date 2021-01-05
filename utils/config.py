import subprocess
import logging
import argparse
logger = logging.getLogger()
import os
home = os.path.expanduser(".")
RAWDATA_DIR = os.path.join(home,"rawdata")
NEWDATA_DIR = os.path.join(home,"newdata")
# EMBED_DIR = os.path.join("vecs")
# EMBED_DIR = "/home/mobtgzhang/vecs"
EMBED_DIR = ""
MODEL_DIR = os.path.join(home,"data","models","checkpoints","logs")
# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'embedding_dim', 'hidden_size','memory_size','sparse','freeze'
}
# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rho', 'eps', 'max_len', 'grad_clipping', 'tune_partial',
    'rnn_padding', 'dropout_rnn', 'dropout_emb'
}
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=16,
                         help='Batch size for training')
    runtime.add_argument('--dev-batch-size', type=int, default=16,
                         help='Batch size during validation/testing')
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--out-dir', type=str, default=NEWDATA_DIR,
                       help='Directory of training/validation/test data')
    files.add_argument('--dataset', type=str,
                       default='bq_corpus',
                       help='Process training dataset:bq_corpus,lcqmc,paws-x-zh')
    files.add_argument('--train-file', type=str,
                       default='train.json',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='dev.json',
                       help='Preprocessed dev file')
    files.add_argument('--test-file', type=str,
                       default='test.json',
                       help='Preprocessed test file')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--words-embedding-file', type=str,
                       default='cc.zh.300.vec',
                       help='Space-separated pretrained embeddings file')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type=bool, default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--document', type='bool', default=False,
                            help='Document words')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='yes',
                         help='The evaluation metric used for model selection: None, exact_match, f1')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=False,
                         help='Sort batches by length for speed')
def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist

    args.train_file = os.path.join(args.out_dir,args.dataset, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.out_dir,args.dataset, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    args.test_file = os.path.join(args.out_dir, args.dataset, args.test_file)
    if not os.path.isfile(args.test_file):
        raise IOError('No such file: %s' % args.test_file)
    # Embeddings options
    args.embedding_dim = 300
    if args.words_embedding_file:
        args.words_embedding_file = os.path.join(args.embed_dir,args.words_embedding_file)
        if not os.path.isfile(args.words_embedding_file):
            args.words_embedding_file = None
            logger.warning('No such file: %s, using random embedding!' % args.words_embedding_file)

    # Set model directory
    if not os.path.exists(args.model_dir):
        subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.words_embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args
def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('Reader Model Architecture')
    model.add_argument('--model-type', type=str, default='treelstm',
                       help='Model architecture type: treelstm, treegru, bilstm,grulstm,lstm,gru')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--memory-size', type=int, default=100,
                       help='memory size of RNN units')
    model.add_argument('--hidden-size', type=int, default=50,
                       help='Hidden size of similarity units')
    model.add_argument('--freeze', type=bool, default=False,
                       help='wether use fixed embeddings.')
    model.add_argument('--sparse', type=bool, default=False,
                       help='If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.')
    # Optimization details
    optim = parser.add_argument_group('Reader Optimization')
    optim.add_argument('--dropout-emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout-rnn', type='bool', default=True,
                       help='Whether to dropout the RNN output')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd, adamax, adadelta')
    optim.add_argument('--learning-rate', type=float, default=1.0,
                       help='Learning rate for sgd, adadelta')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--rho', type=float, default=0.95,
                       help='Rho for adadelta')
    optim.add_argument('--eps', type=float, default=1e-6,
                       help='Eps for adadelta')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune-partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--rnn-padding', type='bool', default=False,
                       help='Explicitly account for padding in RNN encoding')
    optim.add_argument('--max-len', type=int, default=15,
                       help='The max span allowed during decoding')
def override_model_args(old_args, new_args):
    """Set args to new parameters.

    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.

    We keep the new optimation, but leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)

def get_model_args(args):
    """Filter args for model ones.

    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER
    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)