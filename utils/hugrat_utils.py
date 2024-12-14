import tensorflow as tf
import numpy as np
from datasets import load_dataset
import os
import random
from data_IO import make_a_rawhugtokgen, make_a_rando_evenclass#make_a_rawhugtokgen_classwise,
import sys
sys.path.append('../layers/')
#############################################
from typing import Union,List
def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    THIS COMES FROM
    *************************
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_utils.py
    **************************

    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class Hug_Encoder(tf.keras.Model):
  def __init__(self,args,hugmodel):
    '''
    padding mask happens with attn_mask
    hugmodel is at the DistilBertModel level
    hugmodel has embeddings, transformer
    input to transformer is embedded x
    embeddings as weights which is a list,[[2**15,768],[512,768]]

    LAYER-NORM, careful here
      both mycode and hugging do dropout after embedding
      only hugging does layernorm after embedding
      https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_tf_distilbert.html
    positional encodings are also in the embedding layer for hugging
    '''
    super().__init__()
    self.hugmodel=hugmodel
    self.args=args
    if 'distil' in args['hug_chkpt']:
      self.call = self.call_distil
      #self.call_z = self.call_distil
    elif 'bert' in args['hug_chkpt']:
      self.call = self.call_bert
  def call_distil(self,x,attn_mask,training=False,from_logits=False,z=None,):
      x1 = tf.ones((12),dtype=tf.float32)      
      x = embed_hardway(x,self.args['n_v'],self.hugmodel.embeddings,
                            self.hugmodel.embeddings.position_embeddings,
                           training=training,from_logits=from_logits,
                           z=z)  ## if you don't pass z, it does nothing cuz None
      x,_ = self.hugmodel.transformer(x,training=training,
                                    attn_mask=attn_mask,
                                    head_mask=x1,
                                    output_attentions=False,
                                    output_hidden_states=True,
                                    return_dict=None,
                                    )
      return(x)
  def call_bert(self,x,attn_mask,training=False,from_logits=False,z=None,past_key_values_length=0,head_mask=None):
      ### based on https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/bert/modeling_tf_bert.py#L133   
      ## which is diff??? than https://huggingface.co/transformers/v4.4.2/_modules/transformers/models/bert/modeling_tf_bert.html
      input_shape = shape_list(x)
      if from_logits:
        batch_size, seq_length, _ = input_shape
      else:
        batch_size, seq_length = input_shape    
      ###x1 = tf.ones((12),dtype=tf.float32)
      x = embed_hardway_bert(x,self.args['n_v'],self.hugmodel.embeddings,
                            self.hugmodel.embeddings.position_embeddings,
                           training=training,from_logits=from_logits,
                           z=z)  ## if you don't pass z, it does nothing cuz None
      attention_mask_shape = shape_list(attn_mask)
      ###########################      
      mask_seq_length = seq_length + past_key_values_length      
      extended_attention_mask = tf.reshape(attn_mask, (input_shape[0], 1, 1, input_shape[1]))
      extended_attention_mask = tf.cast(extended_attention_mask, dtype=x.dtype)
      one_cst = tf.constant(1.0, dtype=x.dtype)
      ten_thousand_cst = tf.constant(-10000.0, dtype=x.dtype)
      extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
      
      if head_mask is not None:
          raise NotImplementedError
      else:
          head_mask = [None] * self.hugmodel.config.num_hidden_layers

      encoder_outputs = self.hugmodel.encoder(
          hidden_states=x,
          attention_mask=extended_attention_mask,
          head_mask=head_mask,
          output_attentions=0,
          output_hidden_states=1,
          return_dict=0,
          training=training,
      )
      sequence_output = encoder_outputs[0]
      return(sequence_output)      

#@title Glue Code to HF
## Glue code between rationale and HuggingFace
def embed_hardway(x,n_v,embeddings,position_embeddings,training,z=None,
          from_logits=True,
          position_ids=None,expand_it=False):
    if not from_logits:#expand_it:
      x = tf.one_hot(x,depth=n_v) ## hardcode!
    ## straight through!!   ## was done here for a while...but moved      
    ##
    inputs_embeds = tf.linalg.matmul(tf.cast(x,dtype=tf.float32),embeddings.weights[0])
    if z is not None:
      inputs_embeds=inputs_embeds*z
      #tf.print('input_embeds', inputs_embeds)
    input_shape = shape_list(inputs_embeds)[:-1]
    if position_ids is None:
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
    position_embeds = tf.gather(params=position_embeddings, indices=position_ids)
    position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
    ##!!!!!!!
    final_embeddings = embeddings.embeddings_sum(inputs=[inputs_embeds, position_embeds])## double check this!!
    ##!!!!!!!!!
    final_embeddings = embeddings.LayerNorm(inputs=final_embeddings)
    final_embeddings = embeddings.dropout(inputs=final_embeddings, training=training)

    return(final_embeddings)


def embed_hardway_bert(x,n_v,embeddings,position_embeddings,training,z=None,
          from_logits=True,
          position_ids=None,expand_it=False,past_key_values_length=0):
    ## html version      
    ## https://huggingface.co/transformers/v4.4.2/_modules/transformers/models/bert/modeling_tf_bert.html
    if not from_logits:#expand_it:
      x = tf.one_hot(x,depth=n_v) ## hardcode!    
    inputs_embeds = tf.linalg.matmul(tf.cast(x,dtype=tf.float32),embeddings.weights[0])
    if z is not None:
      inputs_embeds=inputs_embeds*z
    input_shape = shape_list(inputs_embeds)[:-1]
    token_type_ids = tf.fill(dims=input_shape, value=0)
    if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )
    position_embeds = tf.gather(params=position_embeddings, indices=position_ids)
    position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
    token_type_embeds = tf.gather(params=embeddings.token_type_embeddings, indices=token_type_ids)
    final_embeddings = embeddings.embeddings_sum(inputs=[inputs_embeds, position_embeds, token_type_embeds])
    final_embeddings = embeddings.LayerNorm(inputs=final_embeddings)
    final_embeddings = embeddings.dropout(inputs=final_embeddings, training=training)

    return(final_embeddings)

######################

def get_max_mask(arr,K=1):
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
        full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=arr.shape)

  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st),default_value=0)  
  return(mask)  

#import sys
#sys.path.append('../share/')
#from IO import make_a_hugtokgen
##########################################################

def printnsay(thefile,text):
  print(text)
  with open(thefile,'a') as f:
    f.write(text+'\n')
def justsay(thefile,text):
  with open(thefile,'a') as f:
    f.write(text+'\n')
def set_seeds(theseed):
  np.random.seed(theseed)
  tf.random.set_seed(theseed)  
  random.seed(theseed)
  os.environ['PYTHONHASHSEED']=str(theseed)

#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)

def just_predict_wrap(jpraw,args,model,x,y,train,bsize):   
  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()  
  return(ddict)

def just_predict_wrap2(jpraw,args,model,x,y,train,bsize): 
  (allpreds,allzs,allpkept,allcostgs,allcostes
        ,alllosss,allobjs,
        allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()  
  return(ddict)


def just_predict_fix(jpraw,args,model,x,y,train,bsize): 
  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
  return(ddict)
def just_predict_fix2(jpraw,args,model,x,y,train,bsize): 
  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['z'+str(i)]=allzs[i].numpy()  
  return(ddict)  


# ##########################################################
def load_ratmodel(args,myModel=None,chkptdir=None):
  mdict={} ## keep all the model things in here
  ## get the right ratmodel
  if myModel is None:
    if 'mtype' not in args or args['mtype']=='hug':
      args['mtype']='hug'
      from hugrat_model import myModel, add_default_args
      from hugrat_model import compute_apply_gradients,jpraw
    elif args['mtype']=='cap':
      print('CAAAAAAAAAAAAPPP')
      from hugcap_model import myModel, add_default_args
      from hugcap_model import compute_apply_gradients,jpraw
    elif args['mtype']=='dyn':
      print('DYNNNNNNNNNNNNN')
      from hugdyn_model import myModel, add_default_args
      from hugdyn_model import compute_apply_gradients,jpraw
    elif args['mtype']=='stop':
      from stoprat_model import myModel, add_default_args
      from stoprat_model import compute_apply_gradients,jpraw
    elif args['mtype']=='comp':
      from comprat_model import myModel, add_default_args
      from comprat_model import compute_apply_gradients,jpraw
    else:
      print('bad mtye', args['mtype'])
  args = add_default_args(args,rollrandom=False)
  ## get the huggingface model
  #model_checkpoint = "distilbert-base-uncased"
  if 'distilbert' in args['hug_chkpt'].lower():
    from transformers import DistilBertTokenizer, TFDistilBertModel

    mdict['tokenizer'] = DistilBertTokenizer.from_pretrained(args['hug_chkpt'],
                                      model_max_length=args['max_len'])
    model_gen = TFDistilBertModel.from_pretrained(args['hug_chkpt'],
                    dropout = args['dropout_gen'],attention_dropout = args['dropout_gen'], ## not sure if right
                    qa_dropout = args['dropout_gen'],seq_classif_dropout = args['dropout_gen']) ## not sure if right
    model_enc = TFDistilBertModel.from_pretrained(args['hug_chkpt'],
                    dropout = args['dropout_enc'],attention_dropout = args['dropout_enc'], ## not sure if right
                    qa_dropout = args['dropout_enc'],seq_classif_dropout = args['dropout_enc']) ## not sure if right                  

    ## build the rationale model
    if args['mtype']=='stop':
      if 'stop_list' not in args:
        print('default stop list')
        args['stop_list'] = ['time', 'what', 'about', 'then', 'a', 'so', 'if', 'i', 'where', 's', 't', 'such', 'people', 'know', 'many', 're', 'how', 'film', '[CLS]', 'any', '[SEP]', 'movie', 'much', 'never', 'too', 'think', 'really', 'better', 'the', 'of', 'and', 'in', 'to', 'was', 'is', 'for', 'on', 'that', 'it', 'at', 'had', 'an', 'be', "'", 'or', 'have', ',', 'all', '.', '-', 'who', 'there', 'two', 'no', '?', 'over', 'can', 'some', 'even', '##ly', 'great', 'why', 'story', 'with', 'his', 'by', '##s', 'this', 'are', '(', ')', 'has', '/', 'him', 'would', 'only', 'well', 'say', 'seen', 'you', 'but', 'not', 'their', 'me', 'made', 'your', 'off', 'get', 'because', 'very', 'scenes', 'he', 'from', 'out', 'we', 'other', 'just', '##ing', 'when', 'will', 'did', 'being', 'these', 'way', 'end', 'love', 'does', 'un', '"', 'films', 'acting', 'as', 'her', 'one', 'also', 'been', 'into', 'its', 'more', 'like', 'than', 'while', 'still', 'little', 'character', 'characters', 'up', 'after', 'best', 'don', '!', 'ever', ':', 'm', 'do', 'see', 'man', 'go', 'make', 'plot', 'most', 'real', 'movies', 'watch', 'which', 'were', 'they', 'them', 'life', 've', 'she', 'through', '##y', 'watching', 'should', 'those', 'first', 'back', 'bad', 'my', 'here', 'good', 'scene', 'could', 'something']
      stop_ids = [mdict['tokenizer'].encode(t)[1] for t in args['stop_list']]
      print('\n\nSTOPIDS', stop_ids,'\n\n')
      ## build the rationale model
      mdict['model'] = myModel(args,
              hugmodel_gen=model_gen.distilbert,hugmodel_enc=model_enc.distilbert,
                pad_id=mdict['tokenizer'].pad_token_id,
                cls_id=mdict['tokenizer'].cls_token_id,
                sep_id=mdict['tokenizer'].sep_token_id,
                stop_ids=stop_ids)    
    else:
      mdict['model'] = myModel(args,
      hugmodel_gen=model_gen.distilbert,hugmodel_enc=model_enc.distilbert,
                pad_id=mdict['tokenizer'].pad_token_id,
                cls_id=mdict['tokenizer'].cls_token_id,
                sep_id=mdict['tokenizer'].sep_token_id,
                                          )


  elif 'bert' in args['hug_chkpt'].lower():      
    from transformers import BertTokenizer, TFBertModel
    mdict['tokenizer'] = BertTokenizer.from_pretrained(args['hug_chkpt'],
                                      model_max_length=args['max_len'])
    model_gen = TFBertModel.from_pretrained(args['hug_chkpt'],                    
                    from_pt=True,
                    hidden_dropout_prob=args['dropout_gen'],attention_probs_dropout_prob=args['dropout_gen'],
                    )
    model_enc = TFBertModel.from_pretrained(args['hug_chkpt'],                                                    
                    from_pt=True,
                    hidden_dropout_prob=args['dropout_enc'],attention_probs_dropout_prob=args['dropout_enc'],
                    )

    ## build the rationale model
    if args['mtype']=='stop':
      if 'stop_list' not in args:
        print('default stop list')
        args['stop_list'] = ['time', 'what', 'about', 'then', 'a', 'so', 'if', 'i', 'where', 's', 't', 'such', 'people', 'know', 'many', 're', 'how', 'film', '[CLS]', 'any', '[SEP]', 'movie', 'much', 'never', 'too', 'think', 'really', 'better', 'the', 'of', 'and', 'in', 'to', 'was', 'is', 'for', 'on', 'that', 'it', 'at', 'had', 'an', 'be', "'", 'or', 'have', ',', 'all', '.', '-', 'who', 'there', 'two', 'no', '?', 'over', 'can', 'some', 'even', '##ly', 'great', 'why', 'story', 'with', 'his', 'by', '##s', 'this', 'are', '(', ')', 'has', '/', 'him', 'would', 'only', 'well', 'say', 'seen', 'you', 'but', 'not', 'their', 'me', 'made', 'your', 'off', 'get', 'because', 'very', 'scenes', 'he', 'from', 'out', 'we', 'other', 'just', '##ing', 'when', 'will', 'did', 'being', 'these', 'way', 'end', 'love', 'does', 'un', '"', 'films', 'acting', 'as', 'her', 'one', 'also', 'been', 'into', 'its', 'more', 'like', 'than', 'while', 'still', 'little', 'character', 'characters', 'up', 'after', 'best', 'don', '!', 'ever', ':', 'm', 'do', 'see', 'man', 'go', 'make', 'plot', 'most', 'real', 'movies', 'watch', 'which', 'were', 'they', 'them', 'life', 've', 'she', 'through', '##y', 'watching', 'should', 'those', 'first', 'back', 'bad', 'my', 'here', 'good', 'scene', 'could', 'something']
      stop_ids = [mdict['tokenizer'].encode(t)[1] for t in args['stop_list']]
      print('\n\nSTOPIDS', stop_ids,'\n\n')
      ## build the rationale model
      mdict['model'] = myModel(args,
              hugmodel_gen=model_gen.bert,hugmodel_enc=model_enc.bert,
                pad_id=mdict['tokenizer'].pad_token_id,
                cls_id=mdict['tokenizer'].cls_token_id,
                sep_id=mdict['tokenizer'].sep_token_id,
                stop_ids=stop_ids)    
    else:
      mdict['model'] = myModel(args,
      hugmodel_gen=model_gen.bert,hugmodel_enc=model_enc.bert,
                pad_id=mdict['tokenizer'].pad_token_id,
                cls_id=mdict['tokenizer'].cls_token_id,
                sep_id=mdict['tokenizer'].sep_token_id,
                                          )                    


  ## chkpt stuff ... care about which "model"
  mdict['theckpt'] = tf.train.Checkpoint(step=tf.Variable(1),
                              net=mdict['model'])
  mdict['chkptman'] = tf.train.CheckpointManager(mdict['theckpt'],
                                        args['log_path'],
                                        max_to_keep=1)
  
  if chkptdir is None and len(args['load_chkpt_dir'])>0:
    chkptdir=args['load_chkpt_dir']
  if chkptdir is not None:
    print('LOADING RATMODEL', chkptdir)
    print('can i get out of loading huggingface in this case?')
    mdict['theckpt'].restore(
                tf.train.latest_checkpoint(chkptdir)
                            ).assert_nontrivial_match()
  else:
    print('NO lOAD RATMODEL, load_hugmodel')    
      
  ## setup optimizers
  mdict['optimizers']={}
  mdict['optimizers']['enc'] = tf.keras.optimizers.Adam(learning_rate=args['enc_lr'])
  mdict['optimizers']['gen'] = tf.keras.optimizers.Adam(learning_rate=args['gen_lr']) 
  if args['mtype']=='comp':
    mdict['optimizers']['compent'] = tf.keras.optimizers.Adam(learning_rate=args['enc_lr'])
  return(args,mdict,compute_apply_gradients,jpraw)
##########################################################

##########################################################
def get_dataset(args,datakey,tokenizer,train,classpick=-1):
  if args['data_type']=='hug':
    hugdata = load_dataset(datakey)
    hugdata = hugdata.shuffle()
    ## get a tf dataset object
    if train:
      tfdata = tfdata_from_hugdata(hugdata['train'],tokenizer,
        maxlen=args['max_len'])
    else:
        tfdata = tfdata_from_hugdata(hugdata['test'],tokenizer,
          maxlen=args['max_len'])
  elif 'ratform' in args['data_type']:#args['data_type']=='ratform':
    print('get_data ratform')
    if 'rando' not in args['data_type'] or not train:    
      tfdata = make_a_rawhugtokgen(datakey,
                    args['aspects'],tokenizer,
                    args['max_len'],addstartend=0,binit=1,classpick=classpick)    
    elif 'rando' in args['data_type'] and train:
      print('RANDO')
      tfdata = make_a_rando_evenclass(datakey,
                    args['aspects'],tokenizer,
                    args['max_len'],addstartend=0,binit=1,
                    classpick=classpick)
  return(tfdata)
    

############################################################
## make the dataset into a tensorflow dataset object.....
## this returns a generator that returns a (x,y) sample
def tfdata_from_hugdata(hugdobj,tokenizer,maxlen=256):
  ## this is the generator
  def huggen():
    for thing in hugdobj:
      #print('THING', thing)
      tokd = tokenizer(thing['text'])
      tok_ids = tokd['input_ids']
      if len(tok_ids)>maxlen:
        tok_ids=tok_ids[:maxlen]
      elif len(tok_ids)<maxlen:
        tok_ids = tok_ids+[tokenizer.pad_token_id 
                           for i in range(len(tok_ids),maxlen)]
      label = [float(thing['label'])]
      yield(label,tok_ids)
  dataset = tf.data.Dataset.from_generator(
      huggen,
      ((tf.float32,tf.int32)),
      (1,tf.TensorShape([None])),
      )
  return(dataset)



def get_loss_sc(args,y,allpreds,allzs,allzsum,allzdiff,allpkept,padmask):
  allgs=[];alles=[];alllosss=[];allobjs=[];allsparse=[];allcoherent=[];
  ## generator and encoder loss
  for i in range(len(args['aspects'])):    
    cost_g,cost_e,loss,obj,sparsity,coherent = compute_loss_single(args=args,
                  preds=allpreds[i],
                  y=y[:,i],
                  zsum=allzsum[i],
                  zdiff=allzdiff[i],
                  )     
    allgs.append(cost_g);alles.append(cost_e);alllosss.append(loss);allobjs.append(obj);
    allsparse.append(sparsity);allcoherent.append(coherent);

  pmsum = tf.reduce_sum(padmask,axis=1)

  allovers = [-69] #!!!!
  return(allgs,alles,alllosss,allobjs,allovers,allsparse,allcoherent)


def pred_loss(preds,y): 
  loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
  loss =  tf.reduce_mean(input_tensor=loss_mat,axis=0)
  pred_hard = tf.cast(tf.equal(x=preds, y=tf.reduce_max(preds, -1, keepdims=True)),
                          y.dtype)
  pred_hard = pred_hard[:,1]
  right_or_wrong = tf.cast(tf.equal(x=pred_hard, y=y),
                          y.dtype)
  accuracy = tf.reduce_mean(right_or_wrong)
  return(loss,accuracy)
  

def compute_loss_single(args,preds,y,zsum,zdiff): 
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

  coherent_metric = zdiff
  # loss calculations for show
  #cost_g = sparsity_cost + loss
  cost_g = loss  + args['coherent']*coherent_metric ##+ args['sparsity']*sparsity_metric
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
  sparsity_metric=-69 #!!!!
  return(cost_g,cost_e,loss,obj,sparsity_metric,coherent_metric)
  
def cag_wrap_fix(compute_apply_gradients,args,model,x,y,
              optimizer_gen0,optimizer_enc0,                                         
                            train,bsize):
  (allcostgs,allcostes,allzsum,
   allzdiff,allpkept,allcostovers,allobjs,alllosss,
  allsparse,allcoherent,allflex) = compute_apply_gradients(args,model,x,y,
                                                optimizer_gen0,optimizer_enc0,
                                                train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()  
    #ddict['zsum'+str(i)]=allzsum[i].numpy()  
    #ddict['zdiff'+str(i)]=allzdiff[i].numpy()  
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()

  return(ddict)  