import tensorflow as tf
import numpy as np
import random
from collections import defaultdict


####################################
def make_a_rando_evenclass(fname,aspects,embed_layer,
      maxlen=256,addstartend=0,binit=1,classpick=None,):
  def ratdatagen():        
        with open(fname) as f:   
          lines = list(f.read().split('\n'))
          ## you need to get a fresh one every time to insure each epoch is shuffled
          ################
          random.shuffle(lines) #######!!!!!!!!!!!!!!!!!!!!!!$$$$$$!!!!!!!!!!!!#$@#$@#$@#$@#$@#$         
          print('DATA SHUFFLE')
          ###############
          labtrack=defaultdict(list)
          for i,l in enumerate(lines):
            if '\t' in l:
              labtrack[l.split('\t')[0]].append(i)        
          minlabcount = min([len(labtrack[k]) for k in labtrack])
          labs = sorted(list(labtrack.keys()))
          
          for llii in range(minlabcount):
            for lab in labs:            
              line = lines[labtrack[lab][llii]]
              if '\t' in line:                
                scores=[]                
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])                    
                  if classpick is not None or classpick!=-1:
                    if (classpick==1 and score1==0) or (classpick==0 and score1==1):                      
                      continue
                  scores.append(score1)                     
                text_list = line.split('\t')[1]#.split()
                if len(text_list)> 1 and len(scores)>0:
                  tokd = embed_layer(text_list)
                  tok_ids=tokd['input_ids']
                  #print('tok_ids', len(tok_ids),type(tok_ids),tok_ids )
                  if len(tok_ids)>maxlen:
                    tok_ids=tok_ids[:maxlen]
                  elif len(tok_ids)<maxlen:
                    tok_ids = tok_ids+[embed_layer.pad_token_id 
                                        for i in range(len(tok_ids),maxlen)] 
                  #print('shapes', np.shape(scores), np.shape(tok_ids), np.shape(llii))
                  #print(line)
                  #print(score1, classpick, classpick==1,score1==0)
                  if classpick==-1 or scores[0]==classpick:
                    yield(scores,tok_ids)
                  #yield(scores,tok_ids)
                  
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None])),
     )
  return(dataset)
######################################
def toklen(tokenizer,astr):
  toks = tokenizer(astr)
  toks = toks['input_ids']
  ###print('toks', toks)
  toks =  [t for t in toks if t!=tokenizer.pad_token]
  return(len(toks))

def make_a_fcgen_sorted_track(fname,aspects,embed_layer,
      maxlen=256,addstartend=0,binit=1,classpick=None,keepind=None):
  def ratdatagen():        
        with open(fname) as f:   
          lines = list(f.read().split('\n'))
          #lens = [len(t.split(' ')) for t in lines]   
          print('sort by token length...might be bad')       
          lens = [toklen(embed_layer,l.split('\t')[1]) for l in lines] ## might be a slow down/memory thing
          
          for llii in np.argsort(lens)[::-1]: 
            if llii in keepind:
              line = lines[llii]               
              if '\t' in line:                
                scores=[]                
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])                    
                  if classpick is not None or classpick!=-1:
                    if (classpick==1 and score1==0) or (classpick==0 and score1==1):                      
                      continue
                  scores.append(score1)                     
                text_list = line.split('\t')[1]#.split()
                if len(text_list)> 1 and len(scores)>0:
                  tokd = embed_layer(text_list)
                  tok_ids=tokd['input_ids']
                  #print('tok_ids', len(tok_ids),type(tok_ids),tok_ids )
                  if len(tok_ids)>maxlen:
                    tok_ids=tok_ids[:maxlen]
                  elif len(tok_ids)<maxlen:
                    tok_ids = tok_ids+[embed_layer.pad_token_id 
                                        for i in range(len(tok_ids),maxlen)] 
                  #print('shapes', np.shape(scores), np.shape(tok_ids), np.shape(llii))
                  #print(line)
                  #print(score1, classpick, classpick==1,score1==0)
                  yield(scores,tok_ids,llii)
                  
  keepind=keepind
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32,tf.int32)),
     (tf.TensorShape([len(aspects)]), 
     tf.TensorShape([None]),()),
     )
  return(dataset)

###########################################
def make_a_rawhugtokgen(fname,aspects,embed_layer,maxlen=256,addstartend=0,binit=1,classpick=-1):
  '''
  input is a file where each line has the class first, tab, then raw input next
  the text is not assumed to be preprocessed
  '''
  def ratdatagen():
        with open(fname) as f:        
          for line in f:
            breakline=False
            #print('NEWLINE')
            if '\t' in line:
              #print(line)
              if binit:
                scores=[]
                #print('scores',line.split('\t')[0])
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  if score1>=.6:
                    score1=1.0
                  elif score1<=.4:
                    score1=0.0
                  else:
                    #print('CONTINUE', score1)
                    breakline=True
                    break
                    #continue
                  scores.append(score1) ############### split on .6 only!!!!!!!!
                  #print('SCORES', scores)
              else:
                scores=[]  
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  scores.append(score1) ############### split on .6 only!!!!!!!!
              if breakline:
                continue
              #print('WHY AM I HERE?')    
              text_list = line.split('\t')[1]#.split()
              if len(text_list)> 1:
                tokd = embed_layer(text_list)
                tok_ids=tokd['input_ids']
                #print('tok_ids', len(tok_ids),type(tok_ids),tok_ids )
                if len(tok_ids)>maxlen:
                  tok_ids=tok_ids[:maxlen]
                elif len(tok_ids)<maxlen:
                  tok_ids = tok_ids+[embed_layer.pad_token_id 
                                       for i in range(len(tok_ids),maxlen)]                 
                #print('TOKID', max(tok_ids))
                if classpick==-1 or scores[0]==classpick:
                  yield(scores,tok_ids)
              #else:
                #print('no text!!')
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None])),
     )
  return(dataset)
