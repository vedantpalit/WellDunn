# -*- coding: utf-8 -*-
# ***Important notes***:

# Before any running please upload WellXplain.csv in the path of this code.

# How to run the code:

# In the first cell, initialize the important variables:

# dimension: This dataset has 4 dimensions, hence the dimension value should be kept to 4
# random_state: We tried three different random_state for sampling data: 200, 345, and 546. It sets 200 as a defualt. You can change it in this cell if you want.

# You have to assign your huggingface access token read to access_token_read variable in line 33.

# Run python medAlpaca_experiment.py

# Note that you may need to install few packages and libraries that imported to the code which is available in the environmnet.txt file.

# You are done!
from tqdm import tqdm
from numpy.linalg import svd
from numpy.linalg import matrix_rank
import pickle
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import pandas as pd
import numpy as np
from transformers import utils
utils.logging.set_verbosity_error()
import shutil, sys
import torch

access_token_read = "hf_..." #enter your token here
login(token = access_token_read)

# Parameters and Hyperparameters assigning
dimension= 4 #dimension for explainWD
rand_state = 200 # We tried random_state for sampling data: 200, 345, and 546 (use random_seed)

#training parameters

MAX_LEN = 64
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE= 1e-03
resume_training = True # if True, it will check if there is any previous saved model and will start from that point to train. If false, it will start from epoch 0

print("random_state:",rand_state)
print("batch size:",TRAIN_BATCH_SIZE )
print("Max_len:",MAX_LEN)
print("EPOCHS:",EPOCHS)

data_new=pd.read_csv('WellXplain.csv') #load the WellXplain dataset
 
data=pd.DataFrame()
data['Text']=data_new['Text']
data['Aspect']=data_new['Aspect']
data['Aspect1']=data_new['Aspect']
data['Aspect2']=data_new['Aspect']
data['Aspect3']=data_new['Aspect']
data['Aspect4']=data_new['Aspect']
data['Explanations']=data_new['Explanations']

for i in range(1,5):
  if i!=1:
    data['Aspect1']=data['Aspect1'].replace(i,0)

for i in range(1,5):
  if i==2:
    data['Aspect2']=data['Aspect2'].replace(i,1)
  else:
    data['Aspect2']=data['Aspect2'].replace(i,0)

for i in range(1,5):
  if i==3:
    data['Aspect3']=data['Aspect3'].replace(i,1)
  else:
    data['Aspect3']=data['Aspect3'].replace(i,0)

for i in range(1,5):
  if i==4:
    data['Aspect4']=data['Aspect4'].replace(i,1)
  else:
    data['Aspect4']=data['Aspect4'].replace(i,0)

# Preparing the dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['Text']
        self.targets = self.df[target_List].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }

# Functions for saving and loading the model in the case the training
# is interrupted. In this case, we use these functions start training
# again from last check point.
from pathlib import Path
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min
    

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
    
    #Sigmoid Loss Function set up.
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# train_model is the function to train the model on the training data.
def train_model(valid_loss_min, start_epoch,n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):
  for epoch in tqdm(range(max(0,start_epoch), n_epochs), desc="Epochs"):
    train_loss = 0
    valid_loss = 0

    model.train()
    for batch_idx, data in tqdm(enumerate(training_loader),total=len(training_loader), desc="Training Batches", leave=False):
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs,_ = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    ######################
    # validate the model #
    ######################

    model.eval()

    with torch.no_grad():
      for batch_idx, data in tqdm(enumerate(validation_loader, 0), total=len(validation_loader), desc="Validation Batches", leave=False):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs,_ = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      # calculate average losses
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      print('\n Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch+1,
            train_loss,
            valid_loss
            ))

      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch ,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss
  print("Training finished successfully!")
  return model

def SVD_calculation(data_name):
  last_layer_attentions = []
  for batch_idx, data in enumerate(data_name):
      ids = data['input_ids'].to(device, dtype=torch.long)
      mask = data['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
      outputs, output_with_attention = model(ids, mask, token_type_ids)

      attentions = output_with_attention.attentions[0]

      for sample in attentions:
          last_layer_attentions.append((sample[11]).detach().cpu())  # Detach and move to CPU

      # Clear GPU memory
      del ids, mask, token_type_ids, outputs, output_with_attention
      torch.cuda.empty_cache()

  d=[item.detach().numpy() for item in last_layer_attentions]
  U, S, VT = svd(d)

  return U, S, VT


data=data.astype({'Aspect1':'float', 'Aspect2':'float', 'Aspect3':'float',
       'Aspect4':'float'})
target_List = ['Aspect1','Aspect2','Aspect3','Aspect4']
# Sampling data for training set and validation set
test_size = 0.2
val_df = data.sample(frac=test_size, random_state=rand_state).reset_index (drop=True)
train_df = data.drop (val_df.index).reset_index (drop=True)

tokenizer=AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", use_fast=False,output_attentions=True)

train_dataset=CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset=CustomDataset(val_df,tokenizer,MAX_LEN)

train_data_loader = torch.utils.data.DataLoader (
train_dataset,
shuffle=True,
batch_size=TRAIN_BATCH_SIZE,
num_workers=0
)
val_data_loader = torch.utils.data.DataLoader (
valid_dataset,
shuffle=False,
batch_size=VALID_BATCH_SIZE,
num_workers=0
)

class AlpacaClass(torch.nn.Module):
    def __init__(self):
        super(AlpacaClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")
        self.model = AutoModel.from_pretrained("medalpaca/medalpaca-7b", return_dict=True,output_attentions=True)
        self.dropout = torch.nn.Dropout(0.05)
        self.linear = torch.nn.Linear(4096, dimension)

    def forward(self, input_ids, attn_mask, seg_ids):
        with torch.no_grad():
          output = self.model(
              input_ids=input_ids,
              attention_mask=attn_mask
              # token_type_ids=seg_ids
          )
        output_with_attention = output
        last_hidden_state_mean = torch.mean(output.last_hidden_state, dim=1)
        # print("Shape:",last_hidden_state_mean.shape)
        output_dropout = self.dropout(last_hidden_state_mean)
        output = self.linear(output_dropout)
        return output, output_with_attention

model = AlpacaClass()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
if resume_training and Path("ckpt_path.pt").is_file():
    print("Loading the previous model to resume training ...")
    model, optimizer, start_epoch, valid_loss_min = load_ckp("ckpt_path.pt", model, optimizer)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    start_epoch += 1 
    print("Training has been done till epoch number ",start_epoch)
else:
   valid_loss_min = np.inf
   start_epoch = 0
   print("Training from epoch 0 will start since either you select resume_training=False or there is no previouse saved model.")
model = model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]


model = train_model(valid_loss_min,start_epoch,EPOCHS, train_data_loader, val_data_loader, model, optimizer, "ckpt_path.pt", "best.pt")

final_list=[]
last_layer_attentions = []
token_scores = []
for i in val_df['Text']:
  example = i
  encodings = tokenizer.encode_plus(
      example,
      None,
      add_special_tokens=True,
      max_length=MAX_LEN,
      padding='max_length',
      return_token_type_ids=True,
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
  )
  model.eval()
  with torch.no_grad():
      input_ids = encodings['input_ids'].to(device, dtype=torch.long)
      attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
      output,output_with_attention = model(input_ids, attention_mask, token_type_ids)
      final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()

      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      
      final_list.append(final_output[0])
      attentions = output_with_attention.attentions[11][0][0]
      # Calculate attention scores for each token
      attention_scores = torch.sum(attentions, dim=0)
      # Normalize attention scores
      normalized_scores = attention_scores / torch.sum(attention_scores)
      # Associate each score with its corresponding token
      token_score = {}
      for j in range(len(normalized_scores)):
          token = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))[j]
          token_score[token] = normalized_scores[j].item()

      token_scores.append(token_score)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
top = 4
Correct_samples = np.zeros((len(token_scores)))
for i in range(len(token_scores)):
  mytoken = token_scores[i]
  token_remove = []
  Explanation = (val_df.iloc[i]['Explanations']).split()
  for tok in mytoken.keys():
    if  tok in ['[SEP]', '[CLS]']:
      token_remove.append(tok)
  for item in token_remove:
    del mytoken[item]
  token_sorted = dict(sorted(mytoken.items(), key=lambda x:x[1], reverse=True))

  token_sorted_top = dict(list(token_sorted.items())[0:top])
  # print(token_sorted_top)
  # raise KeyboardInterrupt

  common_token = 0
  for item in Explanation:
    if item in token_sorted_top.keys():
      common_token += 1
  if common_token/len(Explanation)>.5:
    Correct_samples[i] = 1


for i in range(len(val_df)):
  val_list.append(val_df[target_List][i:i+1].values.tolist()[0])

def finalLabels(predicted_list,val_list):
  for i in range(len(predicted_list)):
    indices=np.array(predicted_list[i]).argsort()[::-1][:int(sum(val_list[i]))]
    # argsort()[:-1][:n]
    # print(predicted_list,np.array(predicted_list[i]).argsort()[::-1][:int(sum(val_list[i]))])
    for j in range(len(predicted_list[i])):
      if j in indices:
        predicted_list[i][j]=1.0
      else:
        predicted_list[i][j]=0.0
  return predicted_list

from sklearn.metrics import classification_report #get classification report
label_names = target_List

classification_report_LlAma = classification_report(np.argmax(val_list, axis=1), np.argmax(final_list, axis=1),target_names=label_names)
print('******************Evaluation metrics******************')
print(classification_report_LlAma)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(np.argmax(val_list, axis=1), np.argmax(final_list, axis=1))

print("Accuracy: {}".format(accuracy))

U, S, VT = SVD_calculation(val_data_loader)
print("SVD Rank (validation data):", matrix_rank(S))
U, S, VT = SVD_calculation(train_data_loader)
print("SVD Rank (training data):", matrix_rank(S))

print("")

print('Attention Overlap (AO) Score:', sum(Correct_samples)/len(token_scores))
print('Number of samples with the ground truth explanations:',sum(Correct_samples),'Number of total samples:', len(token_scores))
