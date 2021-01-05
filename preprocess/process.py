from ltp import LTP
import json
import os
from tqdm import tqdm
import logging
logger = logging.getLogger()
def WriteFile(readfile,savefile):
    with open(readfile,"r",encoding="utf-8") as rfp:
        ltp = LTP()
        logger.info("Processing file:%s ." % (readfile))
        with open(savefile,'w',encoding='utf-8') as wfp:
            for row in tqdm(rfp,desc="file %s process"%(readfile)):
                sent1,sent2,value = row.split('\t')
                seg,hid = ltp.seg([sent1,sent2])
                sdp = ltp.sdp(hid,mode='tree')
                pos = ltp.pos(hid)
                value = int(value)
                tmpitem = {
                    'sentence1':[seg[0],pos[0],sdp[0]],
                    'sentence2':[seg[1],pos[1],sdp[1]],
                    'value':value
                }
                jsonline = json.dumps(tmpitem)
                wfp.write(jsonline + "\n")
def WriteTest(readfile,savefile):
    with open(readfile,"r",encoding="utf-8") as rfp:
        ltp = LTP()
        logger.info("Processing file:%s ." % (readfile))
        with open(savefile,'w',encoding='utf-8') as wfp:

            for row in tqdm(rfp,desc="file %s process"%(readfile)):
                sent1,sent2 = row.split('\t')
                seg,hid = ltp.seg([sent1,sent2])
                sdp = ltp.sdp(hid,mode='tree')
                pos = ltp.pos(hid)
                tmpitem = {
                    'sentence1':[seg[0],pos[0],sdp[0]],
                    'sentence2':[seg[1],pos[1],sdp[1]]
                }
                jsonline = json.dumps(tmpitem)
                wfp.write(jsonline + "\n")

def ProcessFile(rootrawpath,rootnewpath):
    paths = ['bq_corpus','lcqmc','paws-x-zh']
    files = ['train', 'dev']
    for pth in paths:
        newpath = os.path.join(rootnewpath,pth)
        if not os.path.exists(newpath):
            logger.info("Making dirs:%s "%(newpath))
            os.makedirs(newpath)
        else:
            logger.info("Exists paths:%s ." % (newpath))
        rawpath = os.path.join(rootrawpath,pth)
        for tfp in files:
            readfile = os.path.join(rawpath,tfp+'.tsv')
            savefile = os.path.join(newpath,tfp+'.json')
            if not os.path.exists(savefile):
                WriteFile(readfile,savefile)
                logger.info("Processed file:%s ,and save the file %s ."%(readfile,savefile))
            else:
                logger.info("Exists file:%s ." % (savefile))
    for pth in paths:
        newpath = os.path.join(rootnewpath,pth)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        else:
            logger.info("Exists paths:%s ." % (newpath))
        rawpath = os.path.join(rootrawpath,pth)
        readfile = os.path.join(rawpath,'test.tsv')
        savefile = os.path.join(newpath,'test.json')
        if not os.path.exists(savefile):
            WriteTest(readfile,savefile)
            logger.info("Processed file:%s ,and save the file %s ." % (readfile, savefile))
        else:
            logger.info("Exists file:%s ." % (savefile))
