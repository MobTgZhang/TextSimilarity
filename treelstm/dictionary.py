import unicodedata
import treelstm.constants as cstword
# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------
class Dictionary(object):
    '''
    用于加载字典
    字符符号表示
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    '''
    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {cstword.PAD: cstword.PAD_WORD,
                        cstword.UNK: cstword.UNK_WORD,
                        cstword.BOS: cstword.BOS_WORD,
                        cstword.EOS: cstword.EOS_WORD}
        self.ind2tok = {cstword.PAD_WORD:cstword.PAD,
                        cstword.UNK_WORD:cstword.UNK,
                        cstword.BOS_WORD:cstword.BOS,
                        cstword.EOS_WORD:cstword.EOS}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, cstword.UNK_WORD)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(cstword.UNK_WORD))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        constdict = {cstword.PAD_WORD,cstword.EOS_WORD,cstword.BOS_WORD,cstword.UNK_WORD}
        tokens = [k for k in self.tok2ind.keys()
                  if k not in constdict]
        return tokens