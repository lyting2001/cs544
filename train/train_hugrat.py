
import os
import pickle
import time
import json

import numpy as np
import json
import argparse
from collections import defaultdict

import tensorflow as tf

import sys
sys.path.append('../utils/')
from hugrat_utils import *
#sys.path.append('../layers/')
#from stoprat_model import *
#############################################

def training_epoch_speed():  
  for by,bx in data_train.batch(args['train_batch'],num_parallel_calls=-1).take(TOTAKE):   
    compute_apply_gradients(args,ratdict['model'],bx,by,
                            ratdict['optimizers'],
                            True,bsize=tf.shape(bx)[0])  
#############################################

def training_epoch():
  cgs=[];ces=[];losses=[];objs=[];pks=[];
  tdict=defaultdict(list)
  ti=0
  #for bx,by in zip(train_bxs,train_bys):
  trbsizes=[]
  for by,bx in data_train.batch(args['train_batch']).take(TOTAKE):
    bsize=np.shape(bx)[0]
    #print('train bsize!!!', bsize, np.shape(bx), np.shape(by))
    trbsizes.append(bsize)
    #print('by', np.mean(by.numpy()))
    ddict=cag_wrap_fix(compute_apply_gradients=compute_apply_gradients,
                    args=args,
                    model=ratdict['model'],
                    x=bx,
                    y=by,
                    optimizers=ratdict['optimizers'],                 
                    train=True,
                    bsize=bsize
                    )
    tdict = update_ldict(tdict,ddict)

    if ti%25==0:
      print(epoch,',', ti)
      printnsay(thefile=args['logfile'],
        text = 'traini:'+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
    else:
      justsay(thefile=args['logfile'],
        text = 'traini:'+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
                            
    ti+=1
    #ti_track+=1
  return(trbsizes,tdict)
#############################################
def dev_epoch():
  print('EVAL', ti_track)
  devdict=defaultdict(list)  
  ## evaluation
  bsizes=[]
  egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
  for by,bx in data_dev.batch(args['eval_batch']).take(TOTAKE):
    #print('dev by', np.mean(by.numpy()))
    bsize = np.shape(bx)[0]
    bsizes.append(bsize)          
    dev_ddict=just_predict_wrap_HERE(jpraw=jpraw,args=args, #!!!!!!!!!!!
                                    model=ratdict['model'],
                                    x=bx,
                                    y=by,
                                    train=tf.constant(False,dtype=tf.bool),
                                    bsize=bsize)        
                                  
    devdict = update_ldict(devdict,dev_ddict)
  return(bsizes,devdict)
  ## log it
#############################################
def checkpoint_logic(gotsparse,firstsparse,besdev_epoch,thebesdev):
  gotsparse=0
  devdeviation = np.max(np.abs([np.dot(bsizes,devdict[x])/np.sum(bsizes) 
                      -args['slevel']
                      for x in devdict if 'pkept' in x]))
  print('devdeviation', devdeviation)

  dev_obj = np.mean([np.dot(bsizes,devdict[x])/np.sum(bsizes)
                      for x in devdict if 'cost_g' in x]) ## gen cost 
  if devdeviation<=args['sparse_margin']:
    if not gotsparse:
      firstsparse=True
    else:
      firstsparse=False ## only flips to false if its sparse and previously sparse
    gotsparse=1 
  
    
  thetolerance = 0          
  if (thebesdev-dev_obj)>thetolerance:
    betteracc=1
  else:
    betteracc=0

  # if (
  #     (gotsparse and firstsparse)    or
  #     (gotsparse and betteracc)  or
  #     (not gotsparse and betteracc)
  #     ):  
  gotchkpt=0
  if gotsparse and betteracc:
    besdev_epoch = epoch
    thebesdev = dev_obj
    print('NEW BEST!!', thebesdev)
    if args['dosave']:   
      gotchkpt=1   
      save_path = ratdict['chkptman'].save()
      print('saved bes to', save_path)  # put logfile in right place
  return(gotsparse,firstsparse,betteracc,besdev_epoch,thebesdev,gotchkpt)

#############################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  targs = parser.parse_args()  
  ## args
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)
  args = args['rationale']
  print('ARGS\n\n',args,'\n\n')
  #######
  # make the logpath
  if not os.path.exists(args['log_path']):
      os.makedirs(args['log_path'])
  
  print('ARGS\n\n',args,'\n\n')
  
  ## set random seed
  set_seeds(args['rand_seed'])  
  ## load model
  args,ratdict,compute_apply_gradients,jpraw = load_ratmodel(args)
  ## get data huggingface
  data_train = get_dataset(args,args['train_file'],ratdict['tokenizer'],train=True)
  data_dev = get_dataset(args,args['dev_file'],ratdict['tokenizer'],train=False)
  ## training loop stuff
  besdev_epoch=0
  ti_track=0
  thebesdev = np.inf
  gotsparse=False;firstsparse=False;
  
  if 'TOTAKE' in args:
    TOTAKE=args['TOTAKE']
    print('TOTAKE', TOTAKE)
  else:
    TOTAKE=400000000
  print('ARGS\n\n',args,'\n\n')
  with open(args['log_path']+'config.json','w') as f:
      json.dump(args,f,indent=2)
  
  if args['mtype']=='dyn':
    just_predict_wrap_HERE = just_predict_wrap2
  else:
    just_predict_wrap_HERE = just_predict_wrap
  for epoch in range(args['TOTAKE']):
    ## training
    training_epoch_speed()
    ## dev check
    bsizes,devdict = dev_epoch()
    ## checkpoint logic                                  
    gotsparse,firstsparse,betteracc,besdev_epoch,thebesdev,gotchkpt = checkpoint_logic(
                            gotsparse,firstsparse,besdev_epoch,thebesdev)
    ## log epoch
    printnsay(thefile=args['logfile'],
        text = 'epoch:{:.0f}'.format(epoch) + ','        
        +','.join(['dev_{}:{:.5f}'.format(
              x,np.dot(bsizes,devdict[x])/np.sum(bsizes)) 
              for x in devdict])+',chkpt:'+str(gotchkpt))        
    
    if epoch>=args['abs_max_epoch']-1: ## -1 becuase first epoch is zero!
        printnsay(thefile = args['logfile'],
                text = 'BROKE!! epoch '+str(epoch))
        print('DONE EPOCHS', epoch)
        break
        ## track the epoch
    ratdict['theckpt'].step.assign_add(1)
