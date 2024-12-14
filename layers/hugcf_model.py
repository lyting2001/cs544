from ctypes import cdll
from unittest.mock import mock_open
import tensorflow as tf
import numpy as np
#import transformers
import random
import os
from transformers import DistilBertTokenizer, TFDistilBertModel, TFDistilBertForMaskedLM
import sys
sys.path.append('../utils/')
from hugrat_utils import embed_hardway,get_max_mask, Hug_Encoder



@tf.function
def jpredict_by1(args,ratmodel,cfmodel,x,y,bsize,train=True):    
  allpreds,z,_,_,_,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]


  z = tf.math.round(z)
  masks = tf.cast(tf.not_equal(x, cfmodel.padding_id),
                      tf.float32,
                      name = 'masks_generator')   
  maskgen = ratmodel.get_gen_mask(x)
  if args['iterdecode']:     
    #msum = tf.reduce_sum(masks,axis= 1)  
    msum = tf.reduce_sum(maskgen,axis= 1)  
    K = tf.cast(tf.math.round(
              1/tf.reduce_mean(1/(msum*args['slevel']),axis=0)),
            dtype=tf.int32)    
    #tf.print('slevel', args['slevel'],K,tf.reduce_mean(zsum/msum))            
    dec_ids = cf_jpredict_dynamic(cfmodel,x,z,masks,K,
                        train=False)
    
  else:  
    x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
    logits = cfmodel(x_cfmasked,masks,training=train,from_logits=False)
    newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z    
    dec_ids = tf.argmax(newlogits,axis=-1)  
  ## YO
  newx = x
  newz = z
  newy = y
  newpreds = allpreds[0]
  return(dec_ids,
      newx,newz,newy,newpreds)  
#########################################################################      

@tf.function
def jpredict_by1_dyn(args,ratmodel,cfmodel,x,y,bsize,train=True):    
  allpreds,z,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]

  z = tf.math.round(z)
  masks = tf.cast(tf.not_equal(x, cfmodel.padding_id),
                      tf.float32,
                      name = 'masks_generator')   
  maskgen = ratmodel.get_gen_mask(x)
  if args['iterdecode']:     
    msum = tf.reduce_sum(maskgen,axis= 1)    
    K = tf.reduce_sum(z)         
    dec_ids = cf_jpredict_dynamic(cfmodel,x,z,masks,K,
                        train=False)
    
  else:  
    x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
    logits = cfmodel(x_cfmasked,masks,training=train,from_logits=False)
    newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z    
    dec_ids = tf.argmax(newlogits,axis=-1)  
  ## YO
  newx = x
  newz = z
  newy = y
  newpreds = allpreds[0]
  return(dec_ids,
      newx,newz,newy,newpreds)  
#########################################################################     


@tf.function
def jpredict_by1_DEBUG(args,ratmodel,cfmodel,x,y,bsize,train=True):    
  allpreds,z,_,_,_,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]
  z = tf.math.round(z)

  masks = tf.cast(tf.not_equal(x, cfmodel.padding_id),
                      tf.float32,
                      name = 'masks_generator')   
  maskgen = ratmodel.get_gen_mask(x)
  if args['iterdecode']:     
    #msum = tf.reduce_sum(masks,axis= 1)  
    msum = tf.reduce_sum(maskgen,axis= 1)  
    K = tf.cast(tf.math.round(
              1/tf.reduce_mean(1/(msum*args['slevel']),axis=0)),
            dtype=tf.int32) 
    K = tf.reduce_min([K,25])
    #tf.print('K',K)
    #tf.print('slevel', args['slevel'])            
    dec_ids = cf_jpredict_dynamic(cfmodel,x,z,masks,K,
                        train=False)
    
  else:  
    x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
    logits = cfmodel(x_cfmasked,masks,training=train,from_logits=False)
    newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z    
    dec_ids = tf.argmax(newlogits,axis=-1)  
  ## YO
  newx = x
  newz = z
  newy = y
  newpreds = allpreds[0]
  return(dec_ids,
      newx,newz,newy,newpreds)  
#########################################################################     

def jpredictby1_wrap(thefn,args,jratd,cfd,x,y,train,bsize):  
  dec_ids,newx,newz,newy,newpreds=thefn(args,jratd['model'],cfd['model'],
                                    x,y,bsize,train)  
  newz = tf.math.round(newz)     #!!!!!!!!!!!!!!!!                               
  rd={}
  rd['dec_ids']=dec_ids.numpy()
  rd['newx']=newx.numpy()
  rd['newy']=newy.numpy()
  rd['newpred']=newpreds.numpy()
  rd['newz']=newz.numpy()
  return(rd)
#########################################################################      
def just_ratpredict_wrap(jpraw,args,model,x,y,train,bsize): 
  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  allzs = tf.math.round(allzs) ##########!!!!!!!!!!!!!!!!!!!!!!!!
  ddict={}
  ddict['newpred']=allpreds[0].numpy()
  ddict['newz']=allzs[0].numpy()
  ddict['obj']=allobjs[0].numpy()
  return(ddict)  

def just_ratpredict_noz_wrap(args,model,x,y,train,bsize):
  allpreds,allzs,_,=model.call_no_z(x,train=train,bsize=bsize, 
                                from_logits=False,temper=0)

  allzs = tf.math.round(allzs) ##########!!!!!!!!!!!!!!!!!!!!!!!!
  ddict={}
  ddict['newpred']=allpreds[0].numpy()
  ddict['newz']=allzs[0].numpy()
  #ddict['obj']=allobjs[0].numpy()
  return(ddict) 


def just_ratpredictdyn_wrap(jpraw,args,model,x,y,train,bsize): 
  (allpreds,allzs,allpkept,allcostgs,allcostes
        ,alllosss,allobjs,
        allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  allzs = tf.math.round(allzs)      
  ddict={}
  ddict['newpred']=allpreds[0].numpy()
  ddict['newz']=allzs[0].numpy()
  ddict['obj']=allobjs[0].numpy()
  return(ddict)   
#########################################################################  
#@tf.function 
def cf_jpredict_dynamic(cfmodel,x,z,padmask,K,
            train=False):  
  zinds = tf.map_fn(lambda x:tf.cast(tf.where(x==1),dtype=tf.int32),
               tf.cast(tf.math.round(z),dtype=tf.int32),
               parallel_iterations=True)  
  zinds = zinds[:,:,0]
  x_cfmasked  = do_cf_mask(x,z,cfmodel.mask_id)  
  for k in range(K):  
    logits = cfmodel(x_cfmasked,padmask,training=train,from_logits=False)
    dec_ids = tf.argmax(logits,axis=-1)  
    ## add it back in    
    keepmask = tf.cast(tf.squeeze(tf.one_hot(tf.gather(zinds,tf.cast(k,dtype=tf.int32),axis=-1),#zinds[:,k],#zi,#
                        depth=tf.shape(x)[1])),dtype=tf.int32)    
    x_cfmasked = x_cfmasked*(1-keepmask) + tf.cast(dec_ids,dtype=tf.int32)*(keepmask)                                
    x_cfmasked = tf.reshape(x_cfmasked,shape=tf.shape(x))
  return(x_cfmasked) 

class HugCF(tf.keras.Model):
  def __init__(self,args,hugmodel,opter, pad_id,mask_id):
    '''
    hugmodel should be TFDistilBertForMaskedLM
    '''
    super(HugCF,self).__init__()    
    self.args = args    
    self.berty = hugmodel  ## hey be careful about passing around variables and copying from hugmodel   
    self.opter=opter
    self.padding_id= pad_id
    self.mask_id=mask_id ##!!!!!!!!!    
    if 'distil' in args['hug_chkpt']:
      self.call = self.call_distil
      self.encoder = Hug_Encoder(args,hugmodel.distilbert)
    elif 'bert' in args['hug_chkpt']:
      self.call = self.call_bert
      self.encoder = Hug_Encoder(args,hugmodel.bert)
  @tf.function(experimental_relax_shapes=True)
  def call_distil(self,x,attn_mask,training=True,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None):
    hidden_states = self.encoder(x,attn_mask=attn_mask,training=training,
                          from_logits=from_logits)
    prediction_logits = self.berty.vocab_transform(hidden_states)  # (bs, seq_length, dim)
    prediction_logits = self.berty.act(prediction_logits)  # (bs, seq_length, dim)
    prediction_logits = self.berty.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    prediction_logits = self.berty.vocab_projector(prediction_logits)    
    return(prediction_logits)
  def call_bert(self,x,attn_mask,training=True,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None):
    hidden_states = self.encoder(x,attn_mask=attn_mask,training=training,
                          from_logits=from_logits) 
    prediction_logits = self.berty.mlm(sequence_output = hidden_states,
                                  training=training)
    return(prediction_logits)

class HugDiscer(tf.keras.Model):
  def __init__(self,args,encoder,opter,pad_id):
    super().__init__()
    self.encoder=encoder
    self.args=args
    self.opter=opter
    self.pad_id=pad_id
    self.outlayer = tf.keras.layers.Dense(2,activation='softmax')
  def call(self,x,padmask=None,train=True,from_logits=False,temper=1e-5,bsize=None):
    print('discer x', np.shape(x))
    enc_out = self.encoder(x,attn_mask=padmask,training=train,
                  from_logits=from_logits)  
    print('enc_out', np.shape(enc_out))                      
    act_out = self.mean_out(enc_out,masks=tf.expand_dims(padmask,axis=-1))
    print('act_out', np.shape(act_out))
    xout = self.outlayer(act_out)
    print('xout', np.shape(xout))
    return(xout)
  def mean_out(self,enc_out,masks):
      rout = enc_out * masks
      rout = tf.reduce_sum(rout,axis=1)/tf.reduce_sum(masks,axis=1) ## 
      return(rout)    

###############################################################    #
###############################################################    #
###############################################################    #
###############################################################    #
###############################################################    #
###############################################################    #
#@title GAN code


def GAN_wrap(args,thefn,cfd,jratd,discer,opt_discer,
            x,y,bsize,
             train,ganlambda):
  (cfloss,cfpredloss,ganloss) = thefn(args,cfd['amodel'],
                                    discer,jratd['amodel'],
                                    x,y,bsize,
                                    cfd['opter'],opt_discer,
                                    train,ganlambda)  
  ddict = {
        'cfloss':cfloss.numpy(),
        'cfpredloss':cfpredloss.numpy(),
        'ganloss':ganloss.numpy(),        
            }   
  return(ddict)

@tf.function    
def jpredict_wdat(args,ratmodel,cfmodel,x,y,train=False,nounk=False,unkid=0,bsize=0):
  allpreds,z,_,_,_,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]
  z = tf.math.round(z)

  #y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
  padmask = 1-tf.cast(tf.math.equal(x, cfmodel.padding_id),dtype=tf.float32)  
  logits = cfmodel(x_cfmasked,padmask,training=train,from_logits=False)
  #logits = tf.nn.softmax(logits,axis=-1)
  #logits_hard = get_max_mask(logits) ## really dont need this cuz of argmax later...
  newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z                                                        
  x_cf = tf.argmax(newlogits,axis=-1)
  allpreds_cf,z_cf,_,_,_,_=ratmodel(x_cf,train=train,bsize=bsize, ##!!!!!!!!!!!!!!! 05/14/21
                                from_logits=False,temper=0,masks=padmask)
  #print('allpreds_cf[0]', np.shape(allpreds_cf[0]))
  cost_d=[0]
  return(cost_d,
         x,allpreds,z,
         x_cf,allpreds_cf,z_cf)   
@tf.function    
def jpredict_wdatdyn(args,ratmodel,cfmodel,x,y,train=False,nounk=False,unkid=0,bsize=0):
  allpreds,z,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]
  z = tf.math.round(z)

  #y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
  padmask = 1-tf.cast(tf.math.equal(x, cfmodel.padding_id),dtype=tf.float32)  
  logits = cfmodel(x_cfmasked,padmask,training=train,from_logits=False)
  #logits = tf.nn.softmax(logits,axis=-1)
  #logits_hard = get_max_mask(logits) ## really dont need this cuz of argmax later...
  newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z                                                        
  x_cf = tf.argmax(newlogits,axis=-1)
  allpreds_cf,z_cf,_,=ratmodel(x_cf,train=train,bsize=bsize, ##!!!!!!!!!!!!!!! 05/14/21
                                from_logits=False,temper=0,masks=padmask)
  #print('allpreds_cf[0]', np.shape(allpreds_cf[0]))
  cost_d=[0]
  return(cost_d,
         x,allpreds,z,
         x_cf,allpreds_cf,z_cf) 

@tf.function    
def jpredict_wdatdynnoz(args,ratmodel,cfmodel,x,y,train=False,nounk=False,unkid=0,bsize=0):
  allpreds,z,_ = ratmodel(x,train=False,bsize=bsize) ## leave on for variety?
  z = z[0]
  z = tf.math.round(z)

  #y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)  
  padmask = 1-tf.cast(tf.math.equal(x, cfmodel.padding_id),dtype=tf.float32)  
  logits = cfmodel(x_cfmasked,padmask,training=train,from_logits=False)
  #logits = tf.nn.softmax(logits,axis=-1)
  #logits_hard = get_max_mask(logits) ## really dont need this cuz of argmax later...
  newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z                                                        
  x_cf = tf.argmax(newlogits,axis=-1)
  allpreds_cf,z_cf,_,=ratmodel.call_no_z(x_cf,train=train,bsize=bsize, ##!!!!!!!!!!!!!!! 05/14/21
                                from_logits=False,temper=0,masks=padmask)
  #print('allpreds_cf[0]', np.shape(allpreds_cf[0]))
  cost_d=[0]
  return(cost_d,
         x,allpreds,z,
         x_cf,allpreds_cf,z_cf)   

           
#######################################################################################################
@tf.function                
def single_GANbyc(args,cfmodel,discer,ratmodel,
               x,y,bsize,opt_cf,opt_discer,train,ganlambda):
  print('x', np.shape(x))             
  #x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  #print('x_nose', np.shape(x_nose))       
  #print('\n\nSINGLE WHERE IS MY MASK?!??!\n\n')      
  allpreds,z,_,_,_,_ = ratmodel(x,train=args['GRtrain'],bsize=bsize) ## leave on for variety?
  z = tf.math.round(z)
  
  if args['toclass']=='positive':
      the_y = tf.zeros_like(y)#y*0+0 ## it will flip in update
  else:
      the_y = tf.ones_like(y)#y*0+1 ## it will flip in update
  
  if train:

    cfloss,cfpredloss,ganloss,flex=GANbyc_update(args,ganlambda,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],y,
                                            bsize,opt_cf,opt_discer,train) 
    
  else:
    cfloss,cfpredloss,ganloss,flex=GAN_jpredict(args,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],y,
                                            bsize,opt_cf,opt_discer,train) 
    
  return(cfloss,cfpredloss,ganloss,flex)    

#######################################################################################################
#######################################################################################################
@tf.function                
def single_GANbycnozdyn(args,cfmodel,discer,ratmodel,
               x,y,bsize,opt_cf,opt_discer,train,ganlambda):
  print('x', np.shape(x))             
  #x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  #print('x_nose', np.shape(x_nose))       
  #print('\n\nSINGLE WHERE IS MY MASK?!??!\n\n')      
  allpreds,z,_ = ratmodel(x,train=args['GRtrain'],bsize=bsize) ## leave on for variety?
  z = tf.math.round(z)
  
  if args['toclass']=='positive':
      the_y = tf.zeros_like(y)#y*0+0 ## it will flip in update
  else:
      the_y = tf.ones_like(y)#y*0+1 ## it will flip in update
  
  if train:

    cfloss,cfpredloss,ganloss,flex=GANbycnozdyn_update(args,ganlambda,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],y,
                                            bsize,opt_cf,opt_discer,train) 
    
    
  return(cfloss,cfpredloss,ganloss,flex)    
#@tf.function
def GANbycnozdyn_update(args,ganlambda,cfmodel,discer,ratmodel,x,y,z,truey,bsize,opt_cf,opt_disc,
      train=True,from_logits=True,temper=1e-5):   
  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ###!!!!!!!!!!!!!!!!!!!!!!!!!!!    
  x_cfmasked = do_cf_mask(x,z,cfmodel.mask_id)    
  padmask = 1-tf.cast(tf.math.equal(x, cfmodel.padding_id),dtype=tf.float32)  
  with tf.GradientTape() as cf_tape, tf.GradientTape() as disc_tape:
    logits = cfmodel(x_cfmasked,padmask,training=train,from_logits=False)  ## not from logits here...    
    logits = tf.nn.softmax(logits,axis=-1)
    
    
    ## straight-through
    logits_hard = get_max_mask(logits)    
    logits = tf.stop_gradient(logits_hard-logits)+logits 
    
    ## grab og data when not z, wasnt on for ganbigbeer0                                                   
    newlogits= grab_og_data(logits,x,z,args['n_v']) ## keep og data when not z                                                      
    ## get RL loss    
    allpreds,_,_=ratmodel.call_no_z(newlogits,train=args['GRtrain'],bsize=bsize, 
                          from_logits=from_logits,temper=temper,masks=padmask)       
    cfpredloss = get_predloss(allpreds[0],y[:,0])  
    ## get GAN loss    
    x_hot = tf.one_hot(x,depth=args['n_v'])    
    padmask2 = tf.concat([padmask,padmask],axis=0)                  
    ganx = tf.cast(tf.concat([x_hot,newlogits],axis=0),dtype=tf.float32)
    gany = tf.cast(tf.concat([tf.ones_like(y),tf.zeros_like(y)],axis=0),
              dtype=tf.float32)    
    discpred = discer(ganx,padmask2,train,from_logits,temper)    
    if args['toclass']=='positive':
      gmask = tf.concat([truey[:,0],1-truey[:,0]],axis=0)
    else:
      gmask = tf.concat([1-truey[:,0],truey[:,0]],axis=0)
    ganloss = get_predloss_wmask(discpred,gany[:,0],gmask)
                        
    cfvars = cfmodel.trainable_variables    
    cfloss = args['RLlambda']*cfpredloss - ganlambda*ganloss
    gradientscf = cf_tape.gradient(cfloss,cfvars)

    discvars = discer.trainable_variables
    gradientsdisc = disc_tape.gradient(ganloss/ganlambda,discvars)     
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
    opt_disc.apply_gradients(zip(gradientsdisc,discvars))
    flex=-69
  return(cfloss,cfpredloss,ganloss,flex)
#######################################################################################################


#######################################################################################################
def pad_not_rat(x,logits,z,padid):#,from_logits):
  '''
  replace everything thats not rat with padid
  '''
  #y=tf.cast(y,dtype=tf.int32)
  z=tf.cast(tf.expand_dims(z,axis=-1),dtype=logits.dtype)
  padhot=tf.one_hot(tf.ones_like(x,dtype=tf.int32)*tf.cast(padid,dtype=tf.int32),
                        depth=tf.shape(logits)[-1],dtype=tf.float32)
  print('logits', np.shape(logits), 'z', np.shape(z), 'padhot', np.shape(padhot))
  
  out = logits*z+(1-z)*padhot
  return(out)
  
  
def do_cf_mask(x,z,mask_id):
  '''
  x2 = [keep everything where z is 0 and everything else to zero]
      + [the ind of interest at the z1 spots]
 
  '''
  #z = tf.math.round(z) ## new 06/10/22, gonna move to another place that captures more things
  z=tf.cast(z,dtype=tf.int32)
  x3 = x*(1-z) + mask_id*z 
  return(x3)

def get_predloss(preds,y):
  #print('pred loss, y,preds', np.shape(y), np.shape(preds))
  loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
  #print('loss_mat', np.shape(loss_mat))
  loss =  tf.reduce_mean(input_tensor=loss_mat,axis=0)
  #print('loss', np.shape(loss))
  return(loss)  
def get_predloss_wmask(preds,y,amask):
  #print('pred loss, y,preds', np.shape(y), np.shape(preds))
  loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)


  loss = tf.reduce_sum(input_tensor=loss_mat*amask,axis=0)
  loss = loss/tf.reduce_sum(amask,axis=0)
  #loss =  tf.reduce_mean(input_tensor=loss_mat,axis=0)
  #print('loss', np.shape(loss))
  return(loss)   

def grab_og_data(dlogits,x,z,n_v):
  xhot = tf.one_hot(x,depth=n_v,axis=-1,dtype=tf.float32) 
  z=tf.expand_dims(z,axis=-1)
  #print('graab og data',dlogits.dtype,xhot.dtype,z.dtype)  
  #tf.print('TF graab og data',dlogits.dtype,xhot.dtype,z.dtype)  
  newlogits = z*dlogits + (1-z)*xhot
  return(newlogits)



##################################################################################
def loadmodeldict(args,cftext='cfmodel',mtype='cfp'):
  ## optimizer stuff
  if args['cf_lr'] == -1:
    cflr =  CustomSchedule(args['cf_dmodel'])
  elif args['cf_lr']==-2:
    cflr = CustomScheduleBert(warmup_steps=args['warmup'],peak_lr=args['peak_lr'])
  else:  
    cflr = tf.Variable(args['cf_lr'],dtype=tf.float32)
  anopt = tf.keras.optimizers.Adam(learning_rate=cflr)  
  ## hug stuff
  #args['hug_chkpt'] = "distilbert-base-uncased"
  if 'distilbert' in args['hug_chkpt'].lower():
    from transformers import DistilBertTokenizer, TFDistilBertModel, TFDistilBertForMaskedLM
    tokenizer = DistilBertTokenizer.from_pretrained(args['hug_chkpt'])

    ## our stuff
    if mtype=='cfp':
      ahug = TFDistilBertForMaskedLM.from_pretrained(args['hug_chkpt'])
      amodel = HugCF(args,ahug,opter=anopt,pad_id = tokenizer.pad_token_id,
                  mask_id = tokenizer.mask_token_id)
    elif mtype=='discer':
      model = TFDistilBertModel.from_pretrained(args['discer_chkpt'],
          dropout = args['dropout_cfp'],attention_dropout = args['dropout_cfp'], ## not sure if right
          qa_dropout = args['dropout_cfp'],seq_classif_dropout = args['dropout_cfp']) ## not sure if right                  )
      anencoder=Hug_Encoder(args,model.distilbert)
      amodel = HugDiscer(args,encoder=anencoder,opter=anopt,pad_id = tokenizer.pad_token_id)
    else:
      raise NotImplementedError('createmodeldict is for "cfp" and "discer"')
  elif 'bert' in args['hug_chkpt'].lower():
    #from transformers import DistilBertTokenizer, TFDistilBertModel
    from transformers import BertTokenizer, TFBertModel, TFBertForMaskedLM
    tokenizer = BertTokenizer.from_pretrained(args['hug_chkpt'])    
    print('NO DROPOUT IMPLEMENTED')          
    ## our stuff
    if mtype=='cfp':
      ahug = TFBertForMaskedLM.from_pretrained(args['hug_chkpt'],from_pt=True,
          hidden_dropout_prob=args['dropout_cfp'],attention_probs_dropout_prob=args['dropout_cfp'],
          )
      amodel = HugCF(args,ahug,opter=anopt,pad_id = tokenizer.pad_token_id,
                  mask_id = tokenizer.mask_token_id)
    elif mtype=='discer':
      model = TFBertModel.from_pretrained(args['discer_chkpt'],from_pt=True,
      hidden_dropout_prob=args['dropout_cfp'],attention_probs_dropout_prob=args['dropout_cfp'],
          )
      anencoder=Hug_Encoder(args,model.bert)
      amodel = HugDiscer(args,encoder=anencoder,opter=anopt,pad_id = tokenizer.pad_token_id)
    else:
      raise NotImplementedError('createmodeldict is for "cfp" and "discer"')    
  ## make checkpoint
  theckpt = tf.train.Checkpoint(step=tf.Variable(1),
                              net=amodel)
  if 'chkps_tokeep' in args:
    to_keep=args['chkps_tokeep']    
  else:
    to_keep=1
  chkptman = tf.train.CheckpointManager(theckpt,
                                        args['log_path']+'/'+cftext+'/',
                                        max_to_keep=to_keep)    
  ## load stuff?
  if os.path.exists(args['log_path']+'/'+cftext+'/'+'checkpoint') and not args['DEBUGGING']:
    theckpt.restore(
                tf.train.latest_checkpoint(args['log_path']+'/'+cftext+'/')
                            ).assert_nontrivial_match()
    print('MODEL LOADED FROM CHECKPOINT', args['log_path']+'/'+cftext+'/')
  else:
    print('NO MODEL CHECKPOINT TO LOAD, FRESH MODEL')
  ## put it in a dictionary
  cfd = {}
  cfd['model']=amodel
  cfd['chkptman']=chkptman
  cfd['theckpt'] = theckpt
  cfd['mtype']=mtype
  cfd['tokenizer']=tokenizer
  return(cfd)    
###############################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)                        
     
class CustomScheduleBert(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=10000,down_steps=200000,peak_lr=1e-4):
    super(CustomScheduleBert, self).__init__()

    self.start_lr = 1e-12
    
    self.warmup_steps = warmup_steps
    self.down_steps = down_steps
    self.peak_lr = peak_lr
    
    self.wstep = (peak_lr-self.start_lr)/warmup_steps
    self.dstep = (peak_lr-self.start_lr)/(down_steps-warmup_steps)
    self.tracker=0

  @tf.function
  def __call__(self, step):
    if step<=self.warmup_steps:
      lr = self.start_lr+self.wstep*step
    else:
      lr = self.peak_lr-self.dstep*(step-self.warmup_steps)
    return(lr)                     
                 

#######################################################################################################
def add_default_args(args,rollrandom=True):
  defaultargs = {"log_path":"",
  "load_chkpt_dir":"",
  "train_file":"",
  "dev_file":"",
  "test_file":"",
  "DEBUGGING":False,  
  'GRtrain':False,
  "numclass":1,
  "binarize":1,
  "aspects" : [0],  
  "max_len" :256 ,
  "train_batch":64,
  "eval_batch":64,  
  "cf_lr":1e-4,  
  "slevel":0.10,
  "dropout_cfp":0.0,
  "dosave":1,
  "reload":0,
  #"coherent":-69,
  #"sparsity":-69,
  "checknum":2000,  
  "initialization":"rand_uni",
  "edump":0,
  "classpick":-1,
  "padnotrat":0,  
  "chk_obj_mult":1.0,
  "rep_obj_mult":1.0,
  "lent_obj_mult":1.0,
  "gent_obj_mult":1.0,
  "GANlambda":1,
  "RLlambda":1,  
  "n_v":30522, ##################!!!!!!!! THIS IS NOT GOOD
  "rlsup":0,
  "toclass":'positive',
  'iterdecode':0,
  "logfile" : "logfile.log"}
  newargs = dict(defaultargs)
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed ## this will be overwritten if in args
  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None 
  for k in args:
    newargs[k]=args[k]
  k=0
  for i in range(len(newargs['aspects'])):
    for j in range(i+1,len(newargs['aspects'])):
      k+=1
  if k==0:
    k+=1      
  newargs['oversize']=k    
  if newargs['oversize']==0:
    newargs['oversize']=1
  
  if newargs['toclass']=='positive':
    newargs['toint']=1
  else:
    newargs['toint']=0
  if 'discer_chkpt' not in newargs:
    newargs['discer_chkpt'] = newargs['hug_chkpt']
  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None       
  #if newargs['logfile']=='logfile.log': 
  #    newargs['logfile'] = newargs['log_path']+newargs['logfile']      
  return(newargs)      
        