import json
def bulid_vocabulary(filename):
    vocabulary = set()
    with open(filename,mode = 'r',encoding='utf-8') as rfp:
        for row in rfp:
            tmpitem = json.loads(row)
            for word in tmpitem['sentence1'][0]:
                vocabulary.add(word)
            for word in tmpitem['sentence2'][0]:
                vocabulary.add(word)
    return vocabulary
def saveVocabulary(vocabulary,savefile):
    with open(savefile,mode='w',encoding='utf-8') as wfp:
        for word in vocabulary:
            wfp.write(word+"\n")