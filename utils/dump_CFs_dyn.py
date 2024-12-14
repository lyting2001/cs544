

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


import tensorflow as tf

import sys
sys.path.append('../layers/')
from hugcf_model import (loadmodeldict, add_default_args,
                           jpredictby1_wrap,jpredict_by1_dyn,
                           just_ratpredict_noz_wrap)
# from hugrat_model import myModel as RatModel
sys.path.append('../utils/')
from hugrat_utils import load_ratmodel, set_seeds, get_dataset, printnsay, justsay
from data_IO import make_a_fcgen_sorted_track

#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)
#######################################


def update_fillcounter(xcf,zs,fillcounter):
  #xcf = xcf.numpy()
  #zs = zs.numpy()
  for x,z in zip(xcf,zs):
    arep = [int(ti) for ti,zi in zip(x,z) if zi>=.5]
    fillcounter.update(arep)
  return(fillcounter)
#######################################

def calc_ent_from_counts(fillcounter):
  '''
  my understanding of scipy.stats.entropy is that it will ignore zero probability
  so entropy([.33,.33,.33,0,0,0,0])=entropy([.33,.33,.33])
  this function does the later
  '''
  p0 = np.array(list(fillcounter.values()))
  p0 = p0/np.sum(p0)
  ent = entropy(p0)
  return(ent)
#######################################
    

def dump_loop(jpredict,thecfd):
  bsizes=[];     devdict=defaultdict(list)
  newkeepers=[]
  acc_cfs=[]
  print('flippit!!', flipit)
  fillcounter=Counter()
  with open(dumpfile,'a') as f:
    bi=0
    for ti,(by,bx_true,b_inds) in enumerate(data_train.batch(1).take(TOTAKE)):      
      ###print(ti,by[0])
      bsize=np.shape(bx_true)[0]
      if bsize>0:
        bsizes.append(bsize)    
        t0 = time.time()
        if flipit:
          the_y = 1-by
        else:
          the_y = by
        ddict=jpredictby1_wrap(thefn=jpredict,args=args,jratd=cfd_rat,
            cfd=thecfd,x=bx_true,
            y=the_y, 
            train=False, 
            bsize=bsize                       
            )
        fillcounter = update_fillcounter(ddict['dec_ids'],
                                        ddict['newz'],fillcounter)    
        ddict_cf = just_ratpredict_noz_wrap(args=args['rationale'],
                                    model=cfd_rat['model'],
                                    x=ddict['dec_ids'],
                                    y=by,
                                    train=tf.constant(False,dtype=tf.bool),
                                    bsize=bsize)  

        # ddict_cf=just_ratpredictdyn_wrap(jpraw=jpraw_rat,
        #                             args=args['rationale'],
        #                             model=cfd_rat['model'],
        #                             x=ddict['dec_ids'],
        #                             y=by,
        #                             train=tf.constant(False,dtype=tf.bool),
        #                             bsize=bsize)     
        ddict['cf_pred']=  ddict_cf['newpred']  
        ddict['cf_z']=  None#ddict_cf['newz'] 
        ##
        #print(toclass, sum(bsizes),' cf acc', ddict_cf['obj'], ', ent',
        # calc_ent_from_counts(fillcounter))  
        ## 
        #acc_cfs.append(ddict_cf['obj'])
        dumpdicts = []
        
        for i in range(len(ddict['newy'])):
            dumpdicts.append({
              'y':str(float(ddict['newy'][i])),
              'pred':str(float(ddict['newpred'][i][1])),
              'pred_cf':str(float(ddict['cf_pred'][i][1])),
              'x':thecfd['tokenizer'].decode(ddict['newx'][i]),
              'xtoks':[thecfd['tokenizer'].decode([x]) for x in ddict['newx'][i]],
              'cf':thecfd['tokenizer'].decode([int(t) for t in ddict['dec_ids'][i]]),
              'cftoks':[thecfd['tokenizer'].decode([x]) for x in ddict['dec_ids'][i]],
              'z':[str(float(zi)) for zi in ddict['newz'][i]],
              'z_cf':None,#[str(float(zi)) for zi in ddict['cf_z'][i]],
              'b_ind':str(int(b_inds[i]))
                            })
 
        for j in range(len(dumpdicts)):
            json.dump(dumpdicts[j],f)
            f.write('\n') 
        bi+=1
    #pflipped = 1-len(newkeepers)/np.sum(bsizes)
    #print('PERCENT FLIPPED', pflipped)         
    #print('num considered', np.sum(bsizes))
  #ent  = calc_ent_from_counts(fillcounter)   
 # acc = 1- np.dot(acc_cfs,bsizes)/np.sum(bsizes)
  return(devdict,bsizes,newkeepers)#,acc,ent)    


#######################################
if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('thedir')
  parser.add_argument('dumpfile')
  #parser.add_argument('-randsamp',default=False)
  parser.add_argument('-td',default='train')
  parser.add_argument('-flipit',default='1')
  parser.add_argument('-iterdecode',default='1')
  targs = parser.parse_args()  
  #if targs.randsamp=='1' or targs.randsamp.lower=='true':
  #  print('doing rand samp',targs.randsamp)
  #  dorandsamp=True    
  #else:
  #  print('doing ARGMAX', targs.randsamp)
  #  dorandsamp=False

  if targs.flipit=='1' or targs.flipit.lower=='true':
    print('FLIPPINGIT',targs.flipit)
    flipit=True    
  else:
    print('NOTTTT flippign it', targs.flipit)
    flipit=False

  

  dumpfile=targs.thedir+targs.dumpfile 
  ## args
  with open(targs.thedir+'/config.json','r') as f:
      cstr = f.read()
  args = json.loads(cstr)


  if targs.iterdecode=='1' or targs.iterdecode.lower=='true':
    print('ITERDECDE',targs.flipit)
    args['iterdecode']=1
  else:
    print('GREEDY DECODE', targs.flipit)
    args['iterdecode'] = 0    

  if targs.td=='train':
    thefile = args['train_file']
  elif targs.td=='dev':
    thefile = args['dev_file']
  elif targs.td=='source':
    thefile = args['rationale']['source_file']
  else:
    thefile = targs.td

  print('THE FILE', thefile)


  ## load default args
  args = add_default_args(args,rollrandom=True)
  args['log_path'] = targs.thedir
  args['logfile'] = args['log_path']+args['logfile']
  #args['cfboth_chkpt_dir']=targs.thedir #!!!!!!!!!!!!!!!!!!!!!!!!1


  ## set random seed
  set_seeds(args['rand_seed'])

  ## load models
  args_rat,cfd_rat,_,jpraw_rat = load_ratmodel(args['rationale'])
  args['rationale'] = args_rat
    ## load cfp
  args['slevel']  = args['rationale']['slevel']
  
  ## this goes to original checkpoint
  args0 =  json.loads(open(targs.thedir+'/negative_config.json','r').read())
  args1 =  json.loads(open(targs.thedir+'/positive_config.json','r').read())
  cfd0 = loadmodeldict(args0,cftext='cfmodel0',mtype='cfp')
  cfd1 = loadmodeldict(args1,cftext='cfmodel1',mtype='cfp')


  
  ## training loop stuff
  besdev_epoch=0
  thebesdev = np.inf
  gotsparse=False;firstsparse=False;
  
  
  if 'TOTAKE' in args:
     TOTAKE=args['TOTAKE']
  else:
    TOTAKE=100000
  #TOTAKE=5
  for toclass in [0,1]:
    thecfd = cfd0 if toclass==0 else cfd1    
    keepers=list(range(100000))
    print('\n\n\n')
    print('SLEVEL',args['slevel'], toclass)
    print('\n')
    if flipit:
      the_toclass = 1-toclass
    else:
      the_toclass = toclass
    data_train = make_a_fcgen_sorted_track(
                  thefile,args['aspects'],thecfd['tokenizer'],
                      args['max_len'],addstartend=1,binit=1,
                      classpick=the_toclass,##!!
                      keepind=keepers)
    devdict,bsizes,keepers=dump_loop(jpredict=jpredict_by1_dyn,
                            thecfd=thecfd)
    print('num keepers', len(keepers))
    # with open('./results_check.txt','a') as f:
    #   f.write(
    #   targs.flipit+targs.iterdecode+targs.td+' '+targs.thedir+' '+str(toclass)+': {:4f} {:4f}'.format(acc,ent)+'\n')
