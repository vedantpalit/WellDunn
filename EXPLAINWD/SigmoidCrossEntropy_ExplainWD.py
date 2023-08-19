# -*- coding: utf-8 -*-
"""

***Important notes***:

Before any running please upload ExplainWD.csv in the session storage. To do so, you can use the Files section in the left side of this page.

How to run the code:

In the first cell, initialize the important variables:

dimension: This dataset has 4 dimensions, hence the dimension value should be kept to 4
random_state: We tried three different random_state for sampling data: 200, 345, and 546. It sets 200 as a defualt. You can change it in this cell if you want.

1. Run the entire cells for "Installs and utils", "Data Loading and seeds", and "Preparing data" sections

2. Run one of the following models from Metods section:

      1. ERNIE
      2. XLNET
      3. PsychBERT
      4. ClinicalBERT
      5. MentalBERT
      6. BERT
      7. RoBERTa

3. Run the entire cells in the MCC results section

Note that if you are going to run the code again for another method, you must run it from the first section again.

You are done!
"""

dimension= 4
# We tried three different random_state for sampling data: 200, 345, and 546.
# It sets 200 as a defualt. You can change it in this cell if you want.
rand_state = 200


# Do not change the following.
MAX_LEN = 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE= 1e-05
flag=False

"""#Installs and utils

In this section, requirements like numpy, pandas libraries are imported. Also, the pretrained models that we are using in this code are downloaded in this section.

Action required: you may need to log in into hugginface in the second cell, for usage of MentalBERT
"""

!pip install transformers
!pip install SentencePiece
!pip install tensorflow_addons
!pip install bertviz

!huggingface-cli login

import pandas as pd
import numpy as np
from bertviz import model_view
from transformers import utils
utils.logging.set_verbosity_error()
import shutil, sys

import torch

from transformers import AutoModel, AutoTokenizer
from transformers import XLNetModel, XLNetTokenizer

"""#Data Loading

**Action required:**

Before runing this cell, you will need to move ExplainWD.csv file into the path that this code is. On Google colab, you can upload MultiWD.csv file into Session Storage.
"""

# Read the ExplainWD dataset.
data_new=pd.read_csv('ExplainWD.csv')

"""#This portion involves generating one-hot encoded vectors of the dimensions

If a post has dimension as aspect 3, then the one-hot vector would be <0 0 1 0>

If a post has dimension as spect 1, then the one-hot vector would be <1 0 0 0>
"""

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

data

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
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

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

# setting up and running CUDA operations. In case it is not available.
# It changes to cpu.

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

#Sigmoid Loss Function set up.

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# train_model is the function to train the model on the training data.

def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf


  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        #print('yyy epoch', batch_idx)
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs,_ = model(ids, mask, token_type_ids)
        #print(outputs.size())

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)

    print('############# Epoch {}: Training End     #############'.format(epoch))

    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################
    # validate the model #
    ######################

    model.eval()

    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs,_ = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics
      print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }

        # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model

"""#4-label"""

data=data.astype({'Aspect1':'float', 'Aspect2':'float', 'Aspect3':'float',
       'Aspect4':'float'})

target_List = ['Aspect1','Aspect2','Aspect3','Aspect4']

# Sampling data for training set and validation set
test_size = 0.2
val_df = data.sample(frac=test_size, random_state=rand_state).reset_index (drop=True)
train_df = data.drop (val_df.index).reset_index (drop=True)

"""#Method

In these secions, we have provided the code for training models: ERNIE, XLNET, PsychBERT, ClinicalBERT, MentalBERT, BERT, and RoBERTa.

***You should not run all of the code in this section. Instead make sure you run one of them based on your choice.***

###ERNIE
"""

tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-en')

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

class ERNIEClass(torch.nn.Module):
    def __init__(self):
        super(ERNIEClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-en')
        self.ernie_model = AutoModel.from_pretrained('nghuyong/ernie-2.0-en', output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output_dict = self.ernie_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = output_dict.last_hidden_state
        attention_weights = output_dict.attentions
        output_dropout = self.dropout(last_hidden_state[:, -1, :])
        output = self.linear(output_dropout)
        return output, attention_weights

model = ERNIEClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
evaluation=[]
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
report=classification_report(val_list, final_list,target_names=label_names)

print(report)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###XLNET"""

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
flag=True

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

class XLNetClass(torch.nn.Module):
    def __init__(self):
        super(XLNetClass, self).__init__()
        self.xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased', output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output_dict = self.xlnet_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = output_dict.last_hidden_state
        attention_weights = output_dict.attentions
        output_dropout = self.dropout(last_hidden_state[:, -1, :])
        output = self.linear(output_dropout)
        return output, attention_weights[-1]

model = XLNetClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      final_list.append(final_output[0])

val_list=[]

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###PsychBERT"""

tokenizer=AutoTokenizer.from_pretrained('mnaylor/psychbert-cased')

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

class PsychBERTClass(torch.nn.Module):
    def __init__(self):
        super(PsychBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attn_mask, seg_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output, output_with_attention


model = PsychBERTClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention.attentions,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###ClinicalBERT"""

tokenizer=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

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

class ClinicalBIGBERTClass(torch.nn.Module):
    def __init__(self):
        super(ClinicalBIGBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attn_mask, seg_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.last_hidden_state[:, 0])
        output = self.linear(output_dropout)
        return output, output_with_attention



model = ClinicalBIGBERTClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
for i in val_df['text']:
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention.attentions,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###MentalBERT"""

!huggingface-cli login

tokenizer=AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased',use_auth_token=True)

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

class MentalBERTClass(torch.nn.Module):
    def __init__(self):
        super(MentalBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased')
        self.model = AutoModel.from_pretrained('mental/mental-bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attn_mask, seg_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.last_hidden_state[:, 0])
        output = self.linear(output_dropout)
        return output, output_with_attention



model = MentalBERTClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
for i in val_df['text']:
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention.attentions,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###BERT"""

tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

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

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension)

    def forward(self, input_ids, attn_mask, seg_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.last_hidden_state[:, 0])
        output = self.linear(output_dropout)
        return output, output_with_attention



model = BERTClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention.attentions,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""###roBERTa"""

tokenizer=AutoTokenizer.from_pretrained("roberta-base")

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

class roBERTaClass(torch.nn.Module):
    def __init__(self):
        super(roBERTaClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModel.from_pretrained("roberta-base", return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension)

    def forward(self, input_ids, attn_mask, seg_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=seg_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.last_hidden_state[:, 0])
        output = self.linear(output_dropout)
        return output, output_with_attention



model = roBERTaClass()
model.to(device)

val_targets=[]
val_outputs=[]
val_list=[]

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

import pickle
file_name = 'ExplainWD_model_'+'RoBERTa'+'_'+'SCE_'+str(dimension)+'_Dimension'+'.pkl'
pickle.dump(model, open(file_name, 'wb'))

final_list=[]
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
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      input_id_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      # raise KeyBoardInteruupt
      #model_view(output_with_attention.attentions,tokens)

      final_list.append(final_output[0])

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

finalLabels(final_list,val_list)

from sklearn.metrics import classification_report
label_names = target_List
print(classification_report(val_list, final_list,target_names=label_names))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_list, final_list)

print("Accuracy: {}".format(accuracy))

"""#Calculation of SVD Ranking"""

last_layer_attentions = []
for batch_idx, data in enumerate(train_data_loader):
    ids = data['input_ids'].to(device, dtype=torch.long)
    mask = data['attention_mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    outputs, output_with_attention = model(ids, mask, token_type_ids)

    if flag==True:
        attentions = output_with_attention  # For XLNET
    elif flag==False:
        attentions = output_with_attention.attentions[0]

    for sample in attentions:
        last_layer_attentions.append((sample[11]).detach().cpu())  # Detach and move to CPU

    # Clear GPU memory
    del ids, mask, token_type_ids, outputs, output_with_attention
    torch.cuda.empty_cache()

from numpy.linalg import svd
from numpy.linalg import matrix_rank

d=[item.detach().numpy() for item in last_layer_attentions]
U, S, VT = svd(d)

print('*********************************************')
# print("Experiment:", dimension)
print("SVD_ranking:", matrix_rank(S))

"""#MCC Results"""

import tensorflow_addons as tfa
from sklearn.metrics import matthews_corrcoef

metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=dimension)
metric.update_state(val_list,final_list)
result = metric.result()
print(result.numpy())
print('target_List:', target_List)

val = pd.DataFrame(val_list, columns = target_List)
fin = pd.DataFrame(final_list, columns = target_List)

from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(val["Aspect1"],fin["Aspect1"]))

from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(val["Aspect2"],fin["Aspect2"]))

from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(val["Aspect3"],fin["Aspect3"]))

from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(val["Aspect4"],fin["Aspect4"]))