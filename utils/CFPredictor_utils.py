import tensorflow as tf
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy
from scipy.special import softmax
import pickle
import json
from hugrat_utils import printnsay

#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)

def get_flipacc(toclass,pred_cf):
  #print('PREDS', pred_cf[:,1]) 
  pred_bin = tf.cast(tf.math.greater_equal(pred_cf[:,1],.5),dtype=tf.int32) 
  #print('pred_bin',pred_bin)
  if toclass=='positive':
    to_y = tf.ones(tf.shape(pred_bin),dtype=tf.int32)
  else:
    to_y = tf.zeros(tf.shape(pred_bin),dtype=tf.int32)
  eqs = tf.math.equal(to_y,pred_bin)
  acc = tf.reduce_sum(tf.cast(eqs,dtype=tf.int32),axis=-1)/tf.shape(pred_cf)[0]
  
  return(acc)

def update_fillcounter(xcf,zs,fillcounter):
  xcf = xcf.numpy()
  zs = zs.numpy()
  maxrep = [];entis=[];
  for x,z in zip(xcf,zs):
    arep = [int(ti) for ti,zi in zip(x,z) if zi>=.5]
    fillcounter.update(arep)
    carep = Counter(arep)
    #maxrep.append(max(carep.values())/len(arep)) ## normalize by num infill toks
    #entis.append(entropy(list(carep.values())))
  return(fillcounter,maxrep,entis)


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


def flip_est_kl(args,cfmodel,ratmodel,data_gener_flip,data_gener_target,train,epoch,
the_jpredict_wdat,todict,cfd_cfp,cfd_rat,do_cf_mask):
  accs=[];bsizes=[]; fillcounter=Counter();#Xfillcounter=Counter();
  numb = 50000000#int(500/args['train_batch'])
  maxreps=[];entis=[];Xentis=[];#kls=[];
  for ii,(y,x) in enumerate(data_gener_flip.batch(args['train_batch']).take(numb)):      
    bsize=np.shape(x)[0]
    bsizes.append(bsize)    
    cost_d,x_nose,allpreds,z,x_nose_cf,allpreds_cf,z_cf = the_jpredict_wdat(
                           args,ratmodel,cfmodel,x,y,train,bsize=bsize)
    accs.append(get_flipacc(args['toclass'],allpreds_cf[0]))
    fillcounter,maxrep,enti = update_fillcounter(x_nose_cf,z,fillcounter)
    fci = Counter()
    fci,_,_ = update_fillcounter(x_nose_cf,z,fci)



    if ii==0:
      dump_it(epoch,args['log_path']+args['toclass']+'.dump',
                    args,cfd_cfp['model'],cfd_rat['model'],x,y,
                    train=False,bsize=bsize,
                    the_jpredict_wdat=the_jpredict_wdat,
                    do_cf_mask=do_cf_mask,cfd_cfp=cfd_cfp)
  flipacc = np.dot(accs,bsizes)/np.sum(bsizes)
  kl =-1
  return(flipacc,kl,fillcounter)



                       
################################################
def dump_it(epoch,dumpfile,args,cfmodel,ratmodel,x,y,train,bsize,
the_jpredict_wdat,do_cf_mask,cfd_cfp):
  ## this is not flipped
  cost_d,x_nose,allpreds,z,x_nose_cf,allpreds_cf,z_cf = the_jpredict_wdat(
                           args,ratmodel,cfmodel,x,y,train,bsize=bsize)
  x_cf_mask = do_cf_mask(x,z,cfmodel.mask_id)                         

  x_nose=x_nose.numpy()                           
  allpreds=allpreds[0].numpy()
  z = z.numpy()
  x_nose_cf=x_nose_cf.numpy()
  allpreds_cf=allpreds_cf[0].numpy()
  z_cf = z_cf[0].numpy()
  y=y.numpy()
  

  with open(dumpfile,'a') as f:
    for i in range(len(x_nose)):
      dumpdict = {
                  'y':str(float(y[i])),
                  'pred':str(float(allpreds[i][1])),
                  'pred_cf':str(float(allpreds_cf[i][1])),              
                  'x':[cfd_cfp['tokenizer'].decode([t]) for t in x[i]],
                'cf':[cfd_cfp['tokenizer'].decode([t]) for t in x_nose_cf[i]],
                'x_mask':[cfd_cfp['tokenizer'].decode([t]) for t in x_cf_mask[i]],
                'z':[str(float(zi)) for zi in z[i]],
                'z_cf':[str(float(zi)) for zi in z_cf[i]],   
                'rowid':str(epoch)+'__'+str(i)           
                }
      json.dump(dumpdict,f)
      f.write('\n')





def chkpt_it(args,modeld):#ratd,cfd):
    modeld['theckpt'].step.assign_add(1)
    save_path = modeld['chkptman'].save()
    print('saved bes to', save_path)
    printnsay(thefile=args['logfile'],text = 'saved chkpt '+str(modeld['theckpt'].step))
        


def get_targetdist(fname,tokenizer,toint):  
  return(None) 