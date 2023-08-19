# -*- coding: utf-8 -*-
"""

# This code is for Gambler loss scenario on MultiWD dataset.
 How to run the code?

  - 1) To run the code first you need to upload the MultiWD.csv into the path this code is.
  - 2) Set classifier_index variable which represents what method (model) are you going to consider.
  - 3) Set target_index varibale which represents what dimension you want to run the code on. It could be 0 for 6-dimension, 1 for 5-dimension, and 2 for 4-dimension.
  - 4) you can set the ran_index with three values which provide three different random_state for data sampling.
  - 5) run the code now.
 note that the possible values for each variable is provided in the line the the variable is assigned.
"""

dimension=6
# We tried three different random_state for sampling data: 200, 345, and 546.
# It sets 200 as a defualt. You can change it in this cell if you want.
rand_state=200

classifier_index = 1  #Models [0:"ERNIE", 1:"BERT", 2:"RoBERTa", 3:"ClinicalBERT", 4:"XLNET", 5:"PsychBERT", 6:"Mental-BERT"]
# Do not change the following.
MAX_LEN = 64
TRAIN_BATCH_SIZE = 32 # for models except XLNET and MentalBERT it is 32, else it is 2 (due to memory issue)
VALID_BATCH_SIZE = 32 # for models except XLNET and MentalBERT it is 32, else it is 2 (due to memory issue)
EPOCHS = 5
LEARNING_RATE= 1e-05

Classifiers = ["nghuyong/ernie-2.0-en", "bert-base-uncased","roberta-base" ,"emilyalsentzer/Bio_ClinicalBERT", "xlnet-base-cased",'nlptown/bert-base-multilingual-uncased-sentiment', "mental/mental-bert-base-uncased"]
Classifiers_Abs = ["ERNIE", "BERT", "RoBERTa", "ClinicalBERT", "XLNET", "PsychBERT", "Mental-BERT"]
TheClassifier = Classifiers[classifier_index]
TheClassifier_Abstract = Classifiers_Abs[classifier_index]

targets_settings =[
   ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational','Emotional'],
   ['Physical', 'Intellectual', 'Social', 'Vocational','Spiritual_Emotional'],
   [ 'Intellectual', 'Social', 'Vocational','Physical_Spiritual_Emotional'],
   [ 'Social', 'Intellectual_Vocational','Physical_Spiritual_Emotional']
]
if dimension==6:
  target_index=0
elif dimension==5:
  target_index=1 # [0: 6-dim, 1:5-dim, 2:4-dim, 3:3-dim] #Set dimension value
elif dimension==4:
  target_index=2
else:
  target_index=3
target_List = targets_settings[target_index]


print('dimension:',dimension)
print('target_list:',target_List)
print('Model:',TheClassifier_Abstract)
print('Random State:',rand_state)

"""#Installs and Utils

In this section, requirements like numpy, pandas libraries are imported. Also, the pretrained models that we are using in this code are downloaded in this section.

Action required: you may need to log in into hugginface for usage of MentalBERT
"""

!pip install transformers
!pip install sentencepiece
!pip install tensorflow_addons
!pip install bertviz

"""#Dataset Prep"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, XLNetTokenizer,XLNetModel
tokenizer = AutoTokenizer.from_pretrained(TheClassifier)
import torch
data=pd.read_csv('MultiWD.csv')

# Preparing the data if we have selected 6-dimension
if dimension ==6:
  data=data.astype({'Spiritual':'float', 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float', 'Emotional':'float'})
  target_List = ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational','Emotional']
# Preparing the data if we have selected 5-dimension
elif dimension == 5:
  data=data.astype({'Spiritual':'int32', 'Physical':'int32', 'Intellectual':'int32', 'Social':'int32', 'Vocational':'int32', 'Emotional':'int32'})
  data ['Spiritual or Emotional'] = data.Spiritual | data.Emotional
  data = data.drop(['Spiritual', 'Emotional'], axis=1)
  data=data.astype({ 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float',
        'Spiritual or Emotional':'float'})
  target_List = ['Physical', 'Intellectual', 'Social', 'Vocational','Spiritual or Emotional']
# Preparing the data if we have selected 4-dimension
elif dimension == 4:
  data=data.astype({'Spiritual':'int32', 'Physical':'int32', 'Intellectual':'int32', 'Social':'int32', 'Vocational':'int32', 'Emotional':'int32'})
  data ['Spiritual or Emotional'] = data.Spiritual | data.Emotional
  data = data.drop(['Spiritual', 'Emotional'], axis=1)
  data=data.astype({ 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float',
        'Spiritual or Emotional':'float'})

  target_List = ['Physical', 'Intellectual', 'Social', 'Vocational','Spiritual or Emotional']
  data=data.astype({ 'Physical':'int32', 'Intellectual':'int32', 'Social':'int32', 'Vocational':'int32','Spiritual or Emotional':'int32'})
  data ['Spiritual or Emotional or Physical'] = data ['Spiritual or Emotional'] | data['Physical']
  target_List = ['Physical', 'Intellectual', 'Social', 'Vocational','Spiritual or Emotional']
  # data=data.astype({ 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float','Spiritual or Emotional':'float'})

  data=data.astype({'Spiritual or Emotional':'int32', 'Physical':'int32', 'Intellectual':'int32', 'Social':'int32', 'Vocational':'int32', })

  data = data.drop(['Physical'], axis=1)
  data=data.drop(['Spiritual or Emotional'], axis=1)
  data=data.astype({  'Intellectual':'float', 'Social':'float', 'Vocational':'float','Spiritual or Emotional or Physical':'float'})
  target_List = ['Intellectual', 'Social', 'Vocational','Spiritual or Emotional or Physical']

else:
  print("dimension can be 6, 5, or 4. Please make sure you select one of these")

class CustomDataset(torch.utils.data.Dataset):   #Dataset Preparation
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['text']
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

test_size = 0.2
val_df = data.sample(frac=test_size, random_state=rand_state).reset_index (drop=True) #Formation of train and test sets
train_df = data.drop (val_df.index).reset_index (drop=True)
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

"""#Model Selection and Model Save/Load"""

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #switches device to cpu if gpu is unavailable
print(device)

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

if classifier_index==0:
  tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-en')
  class ERNIEClass(torch.nn.Module):
    def __init__(self):
        super(ERNIEClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-en')
        self.ernie_model = AutoModel.from_pretrained('nghuyong/ernie-2.0-en', output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension+1)

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
        return output, attention_weights[-1]

  model = ERNIEClass()
  model.to(device)

if classifier_index==1:
  tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
  class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased',output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension+1)

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


if classifier_index==2:
  tokenizer=AutoTokenizer.from_pretrained("roberta-base")
  class roBERTaClass(torch.nn.Module):
    def __init__(self):
        super(roBERTaClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModel.from_pretrained('roberta-base',output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension+1)

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


if classifier_index==3:
  tokenizer=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
  class ClinicalBIGBERTClass(torch.nn.Module):
    def __init__(self):
        super(ClinicalBIGBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension+1)

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


if classifier_index==4:
  class XLNETClass(torch.nn.Module):
    def __init__(self):
        super(XLNETClass, self).__init__()
        self.xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased", output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension + 1)

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

  model = XLNETClass()
  model.to(device)

elif classifier_index==5:
  tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
  class PsychBERTClass(torch.nn.Module):
    def __init__(self):
        super(PsychBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment',output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768,dimension+1)

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



elif classifier_index==6:
  !huggingface-cli login
  tokenizer=AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased',use_auth_token=True)
  class MentalBERTClass(torch.nn.Module):
    def __init__(self):
        super(MentalBERTClass, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased')
        self.model = AutoModel.from_pretrained('mental/mental-bert-base-uncased',output_hidden_states=True, output_attentions=True, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension+1)

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

"""#Loss Function - Gamblers"""

def loss_fn(m_outputs, targets):
        reward = dimension

        tensor_temp = torch.zeros(32,dtype=torch.float)
        tensor_temp.to(device)
        outputs = torch.nn.functional.softmax(m_outputs, dim=1,dtype=torch.float)

        outputs, reservation = outputs[:, :-1], outputs[:, -1]

        # gain = torch.gather(outputs, dim=1, index=targets).squeeze()
        # print("targets:",targets)
        # print("outputs:", outputs)
        # raise KeyboardInterrupt
        # return targets, outputs
        gain = torch.einsum("ij, ij -> i", targets.to(torch.float), outputs)

        # doubling_rate = (gain.max() + reservation / reward).log()
        doubling_rate = -torch.log(gain + reservation/reward)
        return  doubling_rate.mean(), reservation

"""#Training"""

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
val_targets=[]
val_outputs=[]


def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf

  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()

    for batch_idx, data in enumerate(training_loader):
        # print(data['input_ids'])
        ids = data['input_ids'].to(device, dtype = torch.long)
        # print(ids)
        # raise KeyboardInterrupt
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs, _ = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss, reservationn = loss_fn(outputs, targets.type(torch.int64))

        # print(outputs)
        # loss2 = loss_fn2(outputs, targets)

        # print("loss gambler: ",loss)
        # print("reservation: ", reservationn)

        # print("loss2 CE: ", loss2)

        # raise KeyboardInterrupt
        # tar, outp = loss_fn(outputs, targets.type(torch.int64))
        # return tar, outp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)

    # print('############# Epoch {}: Training End     #############'.format(epoch))

    # print('############# Epoch {}: Validation Start   #############'.format(epoch))
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
            outputs, _ = model(ids, mask, token_type_ids)

            loss, _ = loss_fn(outputs, targets.type(torch.int64))
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      # print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
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
      # save_ckp(checkpoint, False, checkpoint_path, best_model_path)

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        # save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    # print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model

import shutil, sys

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "ckpt_path/themodel3.pt", "thebestone3.pt")

"""#Calculation of SVD Ranking

This piece of code is meant to calculate the singular value decomposition rank for each of the models
"""

last_layer_attentions = []
for batch_idx, data in enumerate(train_data_loader):
    ids = data['input_ids'].to(device, dtype=torch.long)
    mask = data['attention_mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    outputs, output_with_attention = model(ids, mask, token_type_ids)

    if classifier_index == 4:
        attentions = output_with_attention  # For XLNET
    else:
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

"""#Metrics"""

def finalLabels2(predicted_list,val_list):

  indices=np.array(predicted_list).argsort()[::-1][:int(sum(val_list))]
  for j in range(len(predicted_list)):
    if j in indices:
      predicted_list[j]=1.0
    else:
      predicted_list[j]=0.0
  return predicted_list


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


final_list= []
examples = []
# for i in val_df['text']:
#   example = i
Final_Outputs = []
for i in range(len(val_df)):
  example  = val_df.loc[i]['text']
  target  = (val_df.loc[i][target_List]).tolist()
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
      output, _ = model(input_ids, attention_mask, token_type_ids)
      temp = torch.Tensor(target).type(torch.int64).to(device)
      loss, reservation = loss_fn(output,temp.reshape([1,dimension]))
      final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      # print(final_output[0][:-1], target)
      temp = finalLabels2(final_output[0][:-1],target)
      # print(temp)

      # raise KeyInterruption
      # print("tepm: ",temp)
      # print("target:", target.tolist())
      # print("All: ", temp+target+torch.Tensor.tolist(reservation))

      final_list.append(temp+target+torch.Tensor.tolist(reservation))
      examples.append(example)
      Final_Outputs.append({'text':example,'target':target, 'output':temp, 'reservation':reservation})
      # record

from operator import itemgetter
sorted_final_list = sorted(final_list, key=itemgetter(2*dimension - 1), reverse = False)

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

def get_accuracies(true_labels, predictions):
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
    cm = multilabel_confusion_matrix(true_labels, predictions)
    total_count = np.array(true_labels).shape[0]
    accuracies = []
    # print(np.array(true_labels).shape[1])
    # raise KeyboardInterrupt
    for i in range(np.array(true_labels).shape[1]):
        true_positive_count = np.sum(cm[i,1,1]).item()
        true_negative_count = np.sum(cm[i,0,0]).item()
        accuracy = (true_positive_count + true_negative_count) / total_count
        accuracies.append(accuracy)
    return accuracies


from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
label_names = target_List
for threshold in [1, .95, 0.9, 0.85, 0.8,.75]:
  list_data = sorted_final_list[:round(threshold*len(sorted_final_list)-1)]
  val_list = [list_data[i][dimension:2*dimension] for i in range(len(list_data))]
  prediction = [list_data[i][0:dimension] for i in range(len(list_data))]
  # print(val_list)
  # print(prediction)
  # print(dimension,len(val_list), len(prediction))
  # raise KeyboardInterrupt
  print('############# '+TheClassifier_Abstract+'_Dim'+str(dimension)+'_run'+str(rand_state)+'_threshold'+str(threshold)+'   #############')

  print(classification_report(val_list, prediction,target_names=label_names))

  accuracies = get_accuracies(val_list,prediction)
  accuracies = [round(accuracies[i],2) for i in range(dimension)]
  print("accuracies for each class:",accuracies)

  val = pd.DataFrame(val_list, columns = target_List)
  fin = pd.DataFrame(prediction, columns = target_List)

  print('MCC:')

  for i in range(dimension):
     label = target_List[i]
     print(label, matthews_corrcoef(val[label],fin[label]))
#   print("Physical", matthews_corrcoef(val["Physical"],fin["Physical"]))
#   print("Spiritual", matthews_corrcoef(val["Spiritual"],fin["Spiritual"]))
#   print("Intellectual", matthews_corrcoef(val["Intellectual"],fin["Intellectual"]))
#   print("Social", matthews_corrcoef(val["Social"],fin["Social"]))
#   print("Vocational", matthews_corrcoef(val["Vocational"],fin["Vocational"]))
#   print("Emotional", matthews_corrcoef(val["Emotional"],fin["Emotional"]))

print('Done!!!!!!!!!!!!')