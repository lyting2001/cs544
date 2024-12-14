import tensorflow as tf
import transformers
import numpy as np
import random
import sys
import os
sys.path.append('../utils/')
#from hugrat_utils import get_loss_sc,embed_hardway,embed_hardway_bert, shape_list
from hugrat_utils import get_loss_sc, Hug_Encoder


def z_metrics(allzs,masks,ys):
  zdiffs=[];pkepts=[];pkept0s=[];pkept1s=[];
  for i in range(len(allzs)):
    z = allzs[i]
    y=ys[:,i]    
    msum = tf.reduce_sum(masks,axis=-1)
    zdiff = tf.reduce_mean(tf.reduce_sum(input_tensor=tf.abs(z[:,1:]-z[:,:-1])
          ,axis=1)/msum) 
    #zdiff1 = tf.reduce_mean(y*tf.reduce_sum(input_tensor=tf.abs(z[:,1:]-z[:,:-1])
    #      ,axis=1)/msum) 
    #zdiff0 = tf.reduce_mean((1-y)*tf.reduce_sum(input_tensor=tf.abs(z[:,1:]-z[:,:-1])
    #      ,axis=1)/msum)     
    
    pkept = tf.reduce_mean(tf.reduce_sum(z*masks,axis=1)/msum)    
    #pkept1 = tf.reduce_mean(y*(tf.reduce_sum(z*masks,axis=1)/msum))
    #pkept0 = tf.reduce_mean((1-y)*(tf.reduce_sum(z*masks,axis=1)/msum))
    pkept_in = tf.reduce_sum(z*masks,axis=1)/msum
    pkept1 = tf.reduce_sum(y*pkept_in)/(tf.reduce_sum(y)+1e-10)
    pkept0 = tf.reduce_sum((1-y)*pkept_in)/(tf.reduce_sum(1-y)+1e-10)
    #pkept_check = (tf.reduce_sum(y)*pkept1 + tf.reduce_sum(1-y)*pkept0)/(
    #                          tf.reduce_sum(y)+tf.reduce_sum(1-y)) 
    # ## this checks we get the same thing when breaking it out by class
    #tf.print('y', y,y.dtype)
    #tf.print('pkepts', pkept, pkept_check, pkept1,pkept0)
    zdiffs.append(zdiff);pkepts.append(pkept)
    pkept0s.append(pkept0);pkept1s.append(pkept1)
  return(zdiffs,pkepts,pkept0s,pkept1s)


def get_loss(args,y,allpreds,allzs,genmask):

  allzdiff,allpkept,allpkept0s,allpkept1s = z_metrics(allzs,genmask,y)

  allgs=[];alles=[];alllosss=[];allobjs=[];allsparse=[];allcoherent=[];
  ## generator and encoder loss
  for i in range(len(args['aspects'])):    
    cost_g,cost_e,loss,obj,sparsity,coherent = compute_loss_single_here(args=args,
                  preds=allpreds[i],
                  y=y[:,i],             
                  zdiff=allzdiff[i],
                  pkept0=allpkept0s[i],
                  pkept1=allpkept1s[i]
                  )     
    allgs.append(cost_g);alles.append(cost_e);alllosss.append(loss);allobjs.append(obj);
    allsparse.append(sparsity);allcoherent.append(coherent);
  return(allgs,alles,alllosss,allobjs,allsparse,allcoherent,allpkept)

def compute_loss_single_here(args,preds,y,zdiff,pkept0,pkept1): 
  #y=tf.squeeze(y)
  #preds = tf.squeeze(preds)
  if not args['binarize']:
    loss_mat = (preds-y)**2   
  else:
    ## from logits was turned off for the sigmoid thing....
    if 'lsmooth' not in args or 'lsmooth'==0:
      loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
    else:
      print('\n\nLABEL SMOOTH')
      loss_mat = tf.keras.losses.binary_crossentropy(y_true=
                      tf.one_hot(tf.cast(y,dtype=tf.int32),depth=2),
                                                          y_pred=preds,
                                                  from_logits=0,
                                              label_smoothing=args['lsmooth'])
  

  loss_vec = loss_mat#tf.reduce_mean(input_tensor=loss_mat, axis=1)
  loss =  tf.reduce_mean(input_tensor=loss_vec,axis=0)


  pkept = (tf.reduce_sum(y)*pkept1 + tf.reduce_sum(1-y)*pkept0)/(
                              tf.reduce_sum(y)+tf.reduce_sum(1-y)) 
  sparsity_metric = tf.abs(pkept-args['slevel'])
  #sparsity_metric1 = (pkept0-args['slevel'])**2
  #sparsity_metric0 = (pkept1-args['slevel'])**2
  #sparsity_metric = (tf.reduce_sum(y)*sparsity_metric1 + 
  #                    tf.reduce_sum(1-y)*sparsity_metric0)/(
  #                            tf.reduce_sum(y)+tf.reduce_sum(1-y)) 

  ###sparsity_metric += (pkept0-pkept1)**2 ######!!!!!!!!!!!!!!!! YO
  #tf.print('sparsity metric', sparsity_metric)
  coherent_metric = zdiff
  cost_g = loss  + args['coherent']*coherent_metric + args['sparsity']*sparsity_metric
  cost_e = loss
  if args['binarize']:
    pred_hard = tf.cast(tf.equal(x=preds, y=tf.reduce_max(preds, -1, keepdims=True)),
                           y.dtype)
    pred_hard = pred_hard[:,1]
    right_or_wrong = tf.cast(tf.equal(x=pred_hard, y=y),
                           y.dtype)
    accuracy = tf.reduce_mean(right_or_wrong)
    
    obj = accuracy
  else:
    obj=cost_g
  #sparsity_metric=-69 #!!!!
  return(cost_g,cost_e,loss,obj,sparsity_metric,coherent_metric)

@tf.function#(input_signamture=[])
def compute_apply_gradients_nogentrain(args,model,x,y,
                optimizers,
                train,bsize): 
  with tf.GradientTape() as enc_tape0:
    allpreds,allzs,padmask = model(x,
                                                  train=False,bsize=bsize)    
    allcostgs,allcostes,alllosss,allobjs,allsparse,allcoherent,allpkept = get_loss(args,y,
                                                              allpreds,allzs,
                                                              genmask=padmask)                                               
    all_e_cost = tf.reduce_sum(allcostes)
    all_g_cost = tf.reduce_sum(allcostgs)
                  
  ####################
  evars=model.encoders.trainable_variables + \
        model.outlayers.trainable_variables 
  

  gradientse = enc_tape0.gradient(all_e_cost, evars)  
  optimizers['enc'].apply_gradients(zip(gradientse, 
                                    evars))

  allflex = [0]
  return(allcostgs,allcostes,allpkept,
          allobjs,alllosss,allsparse,allcoherent,allflex)


  
@tf.function
def jpraw(args,model,x,y,train,bsize):
  allpreds,allzs,padmask = model(x,train=train,bsize=bsize)
  #allgs,alles,alllosss,allobjs,allsparse,allcoherent,allpkept
  allcostgs,allcostes,alllosss,allobjs,allsparse,allcoherent,allpkept, = get_loss(args,y,
                                                              allpreds,allzs,
                                                              genmask=padmask)
  vocent = 0#model.get_ratfreq_entropy(x,allzs[0]) ##!!                                                   
  allflex=[vocent]  
  #allcostgs[0]+=args['vocentlam']*(100-vocent)  
  return(allpreds,allzs,allpkept,allcostgs,allcostes
        ,alllosss,allobjs,
        allsparse,allcoherent,allflex)

#@tf.function#(experimental_relax_shapes=True)
@tf.function#(input_signamture=[])
def compute_apply_gradients(args,model,x,y,
                optimizers,
                train,bsize): 
  with tf.GradientTape() as gen_tape0, tf.GradientTape() as enc_tape0:
    allpreds,allzs,padmask = model(x,
                                                  train=False,bsize=bsize)    
    allcostgs,allcostes,alllosss,allobjs,allsparse,allcoherent,allpkept = get_loss(args,y,
                                                              allpreds,allzs,
                                                              genmask=padmask)                                                
    all_e_cost = tf.reduce_sum(allcostes)
    all_g_cost = tf.reduce_sum(allcostgs)
                  
  ####################
  evars=model.encoders.trainable_variables + \
        model.outlayers.trainable_variables 
  

  gradientse = enc_tape0.gradient(all_e_cost, evars)  
  optimizers['enc'].apply_gradients(zip(gradientse, 
                                    evars))

  ####################
  ## apply generator gradients
  gvars = model.generator.trainable_variables

  gradientsg = gen_tape0.gradient(all_g_cost, gvars)
  optimizers['gen'].apply_gradients(zip(gradientsg, 
                                    gvars))


  allflex = [0]
  return(allcostgs,allcostes,allpkept,
          allobjs,alllosss,allsparse,allcoherent,allflex)

#####################################
# def embed_hardway(x,embeddings,training,from_logits):
#   if not from_logits:
#     '''
#     this does layernorm and dropout and all that business
#     '''    
#     return(embeddings(x,training=training))
#   else:    
#     #inputs_embeds = tf.gather(params=self.weight, indices=input_ids) ## og
#     inputs_embeds = tf.linalg.matmul(tf.cast(x,dtype=tf.float32),embeddings.weights[0])
#     input_shape = shape_list(inputs_embeds)[:-1]
#     if position_ids is None:
#         position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
#     position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
#     position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
#     final_embeddings = embeddings.embeddings_sum(inputs=[inputs_embeds, position_embeds])
#     final_embeddings = embeddings.LayerNorm(inputs=final_embeddings)
#     final_embeddings = embeddings.dropout(inputs=final_embeddings, training=training)

#     return(final_embeddings)



###############################
#######    Encoder  #########
###############################
class Encoder_tran(tf.keras.Model):
    def __init__(self, args,hugmodel,
                 embedding_layer=None,thetype=None,numhidden=None):      
        # define the layers here
        super().__init__()
        #self.dropout=tf.Variable(args['dropout'],name='dropout')
        self.args = args#tf.contrib.checkpoint.NoDependency(args)
        #if thetype is None:
        #  thetype = args['etype']
        ##self.emb_layer = elayer
        #self.padding_id = tf.convert_to_tensor(
        #                                      embedding_layer.padding_id,
        #                                      dtype=tf.int32)
        self.nclasses = nclasses = args['numclass'] 
        self.MAXSEQ=args['max_len'] 
       
        if 'enc_act' not in args or args['enc_act']=='mean_out':
          print('enc mean out')
          self.out_act = self.mean_out
        elif args['enc_act']=='max_out':
          print('enc max out')
          self.out_act = self.max_out
        elif args['enc_act']=='cls_out':
          print('enc csl out, ONLY VALID FOR HUGMODELS')
          self.out_act = self.cls_out            
        #if 'mpe' not in args:
        #  args['mpe']=args['n_v']    
        #if 'tdmodel' in args:
        #  thehidden = args['tdmodel']
        #else:
        #  thehidden=args['hidden_dimension']
        self.tran = Hug_Encoder(args,hugmodel)
    #     self.tran = Tran_Encoder(num_layers=args['tnlayers'],
    #                                          d_model=thehidden,
    #                                          num_heads=args['theads'],
    #                                          dff=args['tdff'],
    #                                          input_vocab_size=args['n_v'],
    #                                          maximum_position_encoding=args['mpe'],####!!!!!!!!
    #                                          rate=args['dropout_enc'])  
    def call(self,x,zpred,masks=None,
               training=True,
               dropout=tf.constant(0.0,dtype=tf.float32),
               from_logits=False,temper=1e-5,bsize=None,xyo=None):

        masks = tf.expand_dims(masks,2)
        z = tf.expand_dims(zpred,2)
        if self.args['enc_act']=='cls_out':
          z = tf.concat([tf.expand_dims(z[:,0]*0+1,axis=-1),z[:,1:]],axis=1) ## make sure cls is in the rationale
        enc_out = self.tran.call(x,z=z,attn_mask=masks,
                                   training=training,from_logits=from_logits)
        rout = self.out_act(enc_out,masks)
        
        return(rout) 
    def max_out(self,enc_out,masks):
      rout = enc_out * masks + (1. - masks) * (-1e6)
      rout = tf.reduce_max(rout, axis=1) 
      return(rout)
    def mean_out(self,enc_out,masks):
      rout = enc_out * masks
      rout = tf.reduce_sum(rout,axis=1)/tf.reduce_sum(masks,axis=1) ## 
      return(rout)
    def cls_out(self,enc_out,masks):
      rout = enc_out[:,0] ## take first time step
      return(rout)      
    
        
  
  
########################################################
###############################
#######    Generator  #########
###############################
class Generator2(tf.keras.Model):        
    def __init__(self, args, hugmodel,emblayer=None):
        super().__init__()
        self.args = args
        self.nclasses = args['numclass']
        self.naspects = len(args['aspects'])
        #self.emblayer = emblayer
        # dimensions RCNN
        #n_d = self.args['hidden_dimension'] 
        #n_e = emblayer.n_d 
        #self.padding_id = tf.convert_to_tensor(
        #                                      emblayer.padding_id,
        #                                      dtype=tf.int32)

        # layer list
        self.glayers = []
        #self.zero_states = []
        for i in range(2):   
          if 1:#elif args['gtype']=='trane':
            #if 'mpe' not in args:
            #  args['mpe']=args['n_v']              
            self.glayers.append(Hug_Encoder(args,hugmodel))
            #self.glayers.append(Tran_Encoder(num_layers=args['tnlayers'],
            #                                 d_model=args['hidden_dimension'],
            #                                 num_heads=args['theads'],
            #                                 dff=args['tdff'],
            #                                 input_vocab_size=args['n_v'],
            #                                 maximum_position_encoding=args['mpe'],####!!!!!!!!
            #                                 rate=args['dropout'])) 

        #self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) 

        
        self.fcs = [tf.keras.layers.Dense(2,activation=None) for i in range(self.naspects)]
        #self.dropper = tf.keras.layers.Dropout(rate=args['dropout'])
       
    #@tf.function    #on
    def get_zlogits(self,x,masks,training=True,bsize=None,slevel=None):        
        if slevel is None:
          slevel=self.args['slevel']          
        #tmask = create_padding_mask(x,self.padding_id)
        h_concat = self.glayers[0](x,masks,training =training,from_logits=False)   
        zs=[];#zsums=[];zdiffs=[];pkepts=[];
        for i in range(self.naspects):      
          logits = self.fcs[i](h_concat)
          return(logits)
    def call(self,x,masks,training=True,bsize=None,slevel=None):        
        if slevel is None:
          slevel=self.args['slevel']          
        #tmask = create_padding_mask(x,self.padding_id)
        h_concat = self.glayers[0](x,masks,training =training,from_logits=False)   
        zs=[];#zsums=[];zdiffs=[];pkepts=[];
        for i in range(self.naspects):      
          logits = self.fcs[i](h_concat)
          z = self.zstuff(logits,masks,training,
                                    bsize=bsize,maxlen=self.args['max_len'],
                                    slevel=slevel)#,bvect,masks,masks2) 
          zs.append(z);#zsums.append(zsum);zdiffs.append(zdiff);pkepts.append(pkept);
        return(zs)#,zsums,zdiffs,pkepts)
    #@tf.function    #on
    def call_log(self,x,masks,training=True,bsize=None,slevel=None):        
        if slevel is None:
          slevel=self.args['slevel']  
        #tmask = create_padding_mask(tf.cast(tf.argmax(x,axis=-1),dtype=tf.int32),self.padding_id)                    
        h_concat = self.glayers[0](x,masks,training=training,from_logits=True)   
        zs=[];#zsums=[];zdiffs=[];pkepts=[];
        for i in range(self.naspects):      
          logits = self.fcs[i](h_concat)
          z = self.zstuff(logits,masks,training,
                                    bsize=bsize,maxlen=self.args['max_len'],
                                    slevel = slevel)#!!!!!!!
          zs.append(z);#zsums.append(zsum);zdiffs.append(zdiff);pkepts.append(pkept);
        return(zs)#,zsums,zdiffs,pkepts)


    def getL2loss(self,):
      # get l2 cost for all parameters
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_variables
                   if 'bias' not in v.name.lower() ]) * self.args['l2_reg']
      return(lossL2)

    @tf.function
    def zstuff(self,z,masks,training=False,bsize=None,maxlen=None,
                   slevel=None):
      msum = tf.reduce_sum(masks,axis= 1)
      z =  tf.math.softmax(z,axis=-1) ## i think this helps numerically
      #zpass = z[:,:,1]+ (1. - masks) * (-1e6)      
      z_hard = z[:,:,1]>=z[:,:,0] ## prob of rationale greater than not
      z_hard = tf.cast(z_hard,tf.float32)
      z_hard = z_hard*masks
      z = tf.stop_gradient(z_hard - z[:,:,1]) + z[:,:,1]    
      #zsum = tf.reduce_sum(input_tensor=z,axis=1)/msum  
      ##print('zsum', zsum)    
      #zdiff = tf.reduce_mean(tf.reduce_sum(input_tensor=tf.abs(z[:,1:]-z[:,:-1])
      #      ,axis=1)/msum) 
      #pkept = tf.reduce_mean(tf.reduce_sum(z*masks,axis=1)/msum)    
      #tf.print('pkept', pkept)    
      return(z)#,zsum,zdiff,pkept)      
#@tf.function
def get_top_k_mask(arr,K,bsize=69,maxlen=69):
  '''
  magic from
  https://stackoverflow.com/questions/43294421/
  
  returns a binary array of shape array 
  where the 1s are at the topK values along axis -1
  '''
  values, indices = tf.nn.top_k(arr, k=K, sorted=False)
  temp_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
        tf.shape(arr)[:(arr.get_shape().ndims - 1)]) + [K])], indexing='ij')
  temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
  full_indices = tf.reshape(temp_indices, [-1, arr.get_shape().ndims])
  values = tf.reshape(values, [-1])

  mask_st = tf.SparseTensor(indices=tf.cast(
        full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=[bsize,maxlen])
  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st),default_value=0)  
  return(mask)        
        

######################################################3
class myModel(tf.keras.Model):
  def __init__(self,args,hugmodel_gen,hugmodel_enc,
        pad_id=None,cls_id=None,sep_id=None):
    super().__init__()
    self.args = args
    self.padding_id=pad_id;self.cls_id=cls_id;self.sep_id=sep_id;
    self.naspects=len(args['aspects'])
    self.nclasses = args['numclass']
    self.generator = Generator2(args, hugmodel_gen)#embedding_layer)    
    self.outlayers=[]
    self.encoders=[]
    for a in range(self.naspects):    
      self.outlayers.append(Outlayer(args))
      self.encoders.append(Encoder_tran(args,hugmodel_enc))#embedding_layer))
  #@tf.function 
  def get_pad_mask(self,x):
    return(tf.cast(tf.not_equal(x, self.padding_id),
                        tf.float32,
                        name = 'masks_generator'))
  def get_gen_mask(self,x):
    m = tf.cast(tf.ones_like(x),dtype=tf.float32)
    for tid in [self.padding_id,self.cls_id,self.sep_id]:
        m = m*tf.cast(tf.not_equal(x, tid), tf.float32)  
    return(m)
  def call(self,x,y=None,train=True,bsize=None,from_logits=False,
          temper=1e-5,masks=None,slevel=None):
    train=tf.cast(train,dtype=tf.bool)
    if masks is None:
      genmask = self.get_gen_mask(x)
      encmask = self.get_pad_mask(x)      
    else:
      genmask=masks
      encmask=masks
    ## generate highlights
    if from_logits:
      allzs = self.generator.call_log(x,
                                          masks=genmask,training=train,bsize=bsize,
                                          slevel=slevel)
    else:
      #,allzsum,allzdiff,allpkept
      allzs = self.generator(x,
                                          masks=genmask,training=train,bsize=bsize,
                                          slevel=slevel)

    allpreds=[]
    for a in range(self.naspects): 
      h_final = self.encoders[a](x,allzs[a],encmask,training=train,
                              from_logits=from_logits,temper=temper,bsize=bsize)    
      preds = self.outlayers[a](h_final)
      allpreds.append(preds)              
    #allzsum,allzdiff,allpkept,
    return(allpreds,allzs,encmask)
  def call_no_z(self,x,y=None,train=True,bsize=None,from_logits=False,
          temper=1e-5,masks=None,slevel=None):
    train=tf.cast(train,dtype=tf.bool)
    if masks is None:
      genmask = self.get_gen_mask(x)
      encmask = self.get_pad_mask(x)      
    else:
      genmask=masks
      encmask=masks
    

    allpreds=[];allzs=[];
    for a in range(self.naspects): 
      allzs.append(encmask)
      h_final = self.encoders[a](x,encmask,encmask,training=train,
                              from_logits=from_logits,temper=temper,bsize=bsize)    
      preds = self.outlayers[a](h_final)
      allpreds.append(preds)              
    #allzsum,allzdiff,allpkept,
    return(allpreds,allzs,encmask)          



  def one_hotter(self,inds):
    '''
    returns the counts for an integer at that position
    if the integer is bigger than depth, it is not counted
    '''
    out = tf.one_hot(inds,depth=self.args['n_v'])
    out = tf.reduce_sum(tf.reduce_sum(out,axis=0),axis=0)
    return(out)
    
  def split_and_merge(self,vals_mask,padding_id=int(1e10)):#(values,mask,padding_id=69):
    '''
    this pads with value n_v+1000 which will not be counted in one_hotter
    '''
    out0,out1= tf.dynamic_partition(vals_mask[:,0],tf.cast(vals_mask[:,1],dtype=tf.int32),2)
    return(tf.concat([out1,
                      (out0*0+self.args['n_v']+1000000)],axis=-1)) 

      
    
  def get_ratfreq_entropy(self,x,z):
    '''
    estimate the entropy in the vocab ratfreq
    '''
    xz = tf.concat([tf.expand_dims(tf.cast(x,dtype=tf.int32),axis=-1),
                      tf.expand_dims(tf.cast(z,tf.int32),axis=-1)],
                    axis=-1)  
    inds = tf.map_fn(self.split_and_merge,xz,parallel_iterations=True)
    wcounts = self.one_hotter(inds)
    wfreq = wcounts/tf.reduce_sum(wcounts)
    ent = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(wfreq),wfreq))
    return(ent) ## negative ent because we want to maximize!!!


  
##################################
### OUT LAYER ###################      
##################################
class Outlayer(tf.keras.Model):
  def __init__(self,args):
    super().__init__()
    self.args = args
    if args['binarize']:
      nclass = args['numclass']*2
    else:
      nclass=args['numclass']    
    self.outlayer = tf.keras.layers.Dense(nclass,activation='softmax')   ########## UMMMM SHIIIIT                    
  def call(self,x):
    preds = self.outlayer(x)
    return(preds)
  def getL2loss(self,):
      # get l2 cost for all parameters
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_variables
                   if 'bias' not in v.name.lower() ]) * self.args['l2_reg']
      return(lossL2)    




def add_default_args(args,rollrandom=False):
  defaultargs = {
    "log_path":"",
  "load_chkpt_dir":"",
  "train_file":"",
  "dev_file":"",
  "test_file":"",
  "embedding" :"",
  "numclass":1,
  "binarize":1,
  "evenup":1,
  "aspects" : [0],
  "max_len" :256 , ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  "train_batch":10,
  "eval_batch":10,
  "gen_lr":1e-3,
  "enc_lr":1e-3,
  "slevel":0.10,
  "sparse_margin":0.01,
  "coherent":1.0,
  "sparsity":1.0,
  "enc_act":"cls_out",
  "dropout_gen":0.0,
  "dropout_enc":0.0,
  "dosave":1,
  "reload":0,
  "checknum":2000,  
  "n_v":30522, ##################!!!!!!!! THIS IS NOT GOOD
  "abs_max_epoch":10,
  "logfile" : "logfile.log",
  "conda_env":"hug4"}
  newargs = dict(defaultargs)  
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed ## this will be overwritten if in args  
  for k in args:
    newargs[k]=args[k]  
  k=0
  for i in range(len(newargs['aspects'])):
    for j in range(i+1,len(newargs['aspects'])):
      k+=1      
  if k==0:
    k+=1  
  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None   
  if newargs['logfile']=='logfile.log': 
      newargs['logfile'] = newargs['log_path']+newargs['logfile']                
  if 'source_train' not in newargs:
    newargs['source_train']=newargs['train_file']
  return(newargs)
