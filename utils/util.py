import json
import gensim
from collections import Counter
from treelstm.dictionary import Dictionary
from treelstm.reader import DocReader
from .config import get_model_args
import logging
logger = logging.getLogger(__name__)
def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)
    for w in model.index2word:
        w = Dictionary.normalize(w)
        words.add(w)
    del model
    return words
def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.words_embedding_file:
        logger.info('Restricting to words in %s' % args.words_embedding_file)
        valid_words = index_embedding_words(args.words_embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['sentence1'][0])
        _insert(ex['sentence2'][0])
    return words
def build_word_dict(args, examples):
    """Return a word dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict
def load_data(args,filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    examples = []
    with open(filename) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    return examples
def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    logger.info('-' * 100)
    # Build a dictionary from the data questions + documents (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    word_dict = build_word_dict(args, train_exs + dev_exs )
    logger.info('Num words = %d' % len(word_dict))
    # Initialize model
    model = DocReader(get_model_args(args), word_dict)

    # Load pretrained embeddings for words in dictionary
    if args.words_embedding_file:
        model.load_embeddings(word_dict.tokens(), args.words_embedding_file)
    return model
def top_words(args, examples, word_dict):
    """Count and return the most common words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['sentence1'][0]:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
        for w in ex['sentence2'][0]:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
