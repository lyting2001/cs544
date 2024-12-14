
import os
import pickle
import time
import json
import random
import math


import numpy as np
import json
import argparse
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.special import softmax


import tensorflow as tf

import sys
sys.path.append('../layers/')
from hugcf_model import (loadmodeldict, add_default_args,
                        single_GANbyc,single_GANbycnozdyn,jpredict_wdatdyn, jpredict_wdat,do_cf_mask)
# from hugrat_model import myModel as RatModel
sys.path.append('../utils/')
from hugrat_utils import load_ratmodel, set_seeds, get_dataset, printnsay, justsay
from data_IO import make_a_rawhugtokgen
from CFPredictor_utils import *


    
##############################################################################
def GAN_wrap(args,thefn,#cfd,jratd,discer,opt_discer,
            x,y,bsize,
             train,ganlambda):
  (cfloss,cfpredloss,ganloss,flex) = thefn(args,cfd_cfp['model'],
                                    cfd_discer['model'],cfd_rat['model'],
                                    x,y,bsize,
                                    cfd_cfp['model'].opter,
                                    cfd_discer['model'].opter,
                                    train,ganlambda)  
  ddict = {
        'cfloss':cfloss.numpy(),
        'cfpredloss':cfpredloss.numpy(),
        'ganloss':ganloss.numpy(),        
        'flex':flex.numpy(),        
            }   
  return(ddict)      

    
 #######################################   
def training_loop(cag,thebsize=None):
  bes_dev_obj=-np.inf#np.inf
  NEVERMET=1
  trbsizes=[];    tdict=defaultdict(list);
  if thebsize is None:
    thebsize=args['train_batch']
  tii=0
  for epoch in range(args['abs_max_epoch']): #######!!!!!!!!!!!
    print('EPOCH JS2S', epoch)    
    for ti,(by,bx) in enumerate(data_train.batch(thebsize).take(TOTAKE)):       
      ganlambda=float(args['GANlambda'])
      the_cag = cag
      
      bsize=np.shape(bx)[0]


      

      ## dump at the beginning
      if ti==0 and dodump==1:
        bx0=bx;by0=by;

      if bsize>1:      
        trbsizes.append(bsize)
        ddict=GAN_wrap(thefn=the_cag,
                        args=args,                                              
                        x=bx,
                        y=by,
                        bsize=bsize,
                        train=True,
                        ganlambda=ganlambda                             
                        )

        tdict = update_ldict(tdict,ddict)              
        if tii%CHECKNUM==0:# and tii>0:

          flipacc,kldiv,fillcounter = flip_est_kl(args,cfd_cfp['model'],cfd_rat['model'],
                        freshgen(args),freshgen2(args),False,epoch,
                        the_jpredict_wdat,todict,cfd_cfp,cfd_rat,do_cf_mask)
          ent = calc_ent_from_counts(fillcounter)              
          printnsay(thefile=args['logfile'],
                        text = 'epoch:{:.0f}'.format(epoch) + ','
                        +','.join(['train_{}:{:.5f}'.format(x,np.dot(trbsizes,tdict[x])/np.sum(trbsizes)) 
                                                      for x in tdict]) +','
                        
                        ) 
          #dev_obj = args['lent_obj_mult']*kldiv+args["chk_obj_mult"]*(1-flipacc)
          dev_obj = args['lent_obj_mult']*ent+args["chk_obj_mult"]*(flipacc)
          if dev_obj>bes_dev_obj:
            cond=True
            bes_dev_obj=dev_obj
          else:
            cond=False

          if  cond: 
            print('CHECKPOINTING BABY')
            printnsay(thefile=args['logfile'],
              text = 'chkpting BABEE,  {:.5f}, {:.5f},{:.5f},{:.5f}'.format(
                              flipacc,kldiv,dev_obj,ent      
              ))            
            chkpt_it(args,cfd_cfp)               
          else:
            printnsay(thefile=args['logfile'],
              text = 'skipt chkpt,  {:.5f}, {:.5f},{:.5f},{:.5f}'.format(
                              flipacc,kldiv,dev_obj,ent      
              ))
          ## RESET STUFF!!!
          trbsizes=[];    tdict=defaultdict(list);
        tii+=1          




      #ti_track+=1
  return(tdict,trbsizes)



#######################################
if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  targs = parser.parse_args()  
  #@title setup args
  ## args
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)

  # make the logpath
  time.sleep(random.randrange(1,5))
  if not os.path.exists(args['log_path']):
      os.makedirs(args['log_path'])  

    
  
  args = add_default_args(args,rollrandom=True)
 
  if args['toclass']=='positive':
    cftext='cfmodel1'
    toint=1
  else:
    cftext='cfmodel0'   
    toint=0


  args['logfile'] = args['log_path']+cftext+'_'+args['logfile']

  
  ## load models
  ## load rat model
  #cfd_rat = loaddict_hugmodel(args['rationale'],chkptdir=None)
  args_rat,cfd_rat,_,jpraw_ratonly = load_ratmodel(args['rationale'])
  args['rationale'] = args_rat
  tpath = args_rat['log_path'].replace('finetuned/','')  
  set_seeds(args['rand_seed'])  
  ## load cfp
  cfd_cfp = loadmodeldict(args,cftext,mtype='cfp')
  ## load discriminator
  cfd_discer = loadmodeldict(args,cftext,mtype='discer')


  numlines = sum(1 for line in open(args['train_file']))
  numbatch = math.ceil(numlines/args['train_batch'])
  if numbatch<args['checknum']:
    CHECKNUM=numbatch
  else:
    CHECKNUM=args['checknum']
  if CHECKNUM>args['TOTAKE']:
    CHECKNUM=args['TOTAKE']
  print('CHECKNUM', CHECKNUM)
  ## get data generators
  todict=None
  data_train = get_dataset(args,args['train_file'],cfd_cfp['tokenizer'],train=True)
  data_dev = get_dataset(args,args['dev_file'],cfd_cfp['tokenizer'],train=False)
  freshgen = lambda args: make_a_rawhugtokgen(args['train_file'],args['aspects'],
                            cfd_cfp['tokenizer'],args['max_len'],
                            addstartend=0,binit=1,
                            classpick=1-toint) # flips only
  freshgen2 = lambda args: make_a_rawhugtokgen(args['train_file'],args['aspects'],
                            cfd_cfp['tokenizer'],args['max_len'],
                            addstartend=0,binit=1,
                            classpick=toint) # target only

  ## training loop stuff
  besdev_epoch1=0  
  thebesdev1 = np.inf
  gotsparse1=False;firstsparse1=False;
  

  besdev_epoch0=0
  thebesdev0 = np.inf
  gotsparse0=False;firstsparse0=False;
  
  if 'TOTAKE' in args:
     TOTAKE=args['TOTAKE']
  else:
    TOTAKE=100000

  # save the config file (argfile)
  with open(args['log_path']+'config.json','w') as f:
      json.dump(args,f,indent=2)     
  with open(args['log_path']+args['toclass']+'_config.json','w') as f:
      json.dump(args,f,indent=2)           

  
  dodump=1


  if args['rationale']['mtype']=='dyn':
    print('DYN!!!')
    #cag = single_noGANbycglobaldyn
    cag = single_GANbycnozdyn
    the_jpredict_wdat = jpredict_wdatdyn
  else:
    cag = single_GANbyc#single_GANbyc #!!!!!!!!!!
    the_jpredict_wdat = jpredict_wdat
  #cag_JD=disceronly_GAN
  oganlambda=float(args['GANlambda'])



  
  #cag_JD=disceronly_GAN
  oganlambda=float(args['GANlambda'])




  tdict,trbsizes=training_loop(cag=cag,thebsize=args['train_batch'])
  printnsay(thefile=args['logfile'],text = 'DONE!!!')




