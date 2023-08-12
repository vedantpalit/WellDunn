MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE= 1e-05
rand_state = 345
top = 4

import torch.nn as nn
import torch
import shutil, sys
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

#Classifiers = ["nghuyong/ernie-2.0-en", "bert-base-uncased","roberta-base" ,"emilyalsentzer/Bio_ClinicalBERT", "xlnet-base-cased","nlptown/bert-base-multilingual-uncased-sentiment", "mental/mental-bert-base-uncased"]

TheClassifier =   "mental/mental-bert-base-uncased"

data=pd.read_csv("DataFile.csv")
zero_vector = np.zeros((len(data)))
data['A1'] = zero_vector
data['A2'] = zero_vector
data['A3'] = zero_vector
data['A4'] = zero_vector



tokenizer = AutoTokenizer.from_pretrained(TheClassifier)

for i in range(len(data)):
   indx = int(data.shape[1] - 5 + data.iloc[i]['Aspect'])
   data.iat[i, indx] = 1.0
  #  print("index:", indx,data.iloc[i][indx])
  #  raise KeyboardInterrupt

# print(data.iloc[0][4:8])

# data=data.astype({'Spiritual':'float', 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float',
#        'Emotional':'float'})
indecis_aspect = []
indecis_aspect.append([data[data['Aspect'] == 1].index.tolist()])
indecis_aspect.append([data[data['Aspect'] == 2].index.tolist()])
indecis_aspect.append([data[data['Aspect'] == 3].index.tolist()])
indecis_aspect.append([data[data['Aspect'] == 4].index.tolist()])

# data = data.drop(['Explanations'], axis=1)
data = data.drop(data.columns[0], axis=1)

dimension = 4
target_List = [ 'A1', 'A2', 'A3', 'A4']



def return_tokenized(string):
  inputs = tokenizer.encode_plus(
            string,
            None,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

  tokenized_input = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
        }
  return tokenized_input
  
  
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
      

test_size = 0.2
val_df = data.sample(frac=test_size, random_state=rand_state).reset_index (drop=True)
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

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

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
    # torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        # shutil.copyfile(f_path, best_fpath)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained(TheClassifier,output_attentions=True)# BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output,output_with_attention

model = BERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

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
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        #print('yyy epoch', batch_idx)
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
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model
  
  
model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

final_list=[]
last_layer_attentions = []
token_scores = []
k=0
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
      output, output_attentions = model(input_ids, attention_mask, token_type_ids)
      attentions = output_attentions.attentions[11][0][0]
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

Correct_samples = np.zeros((len(token_scores)))
Correct_samples_No_overlap = np.zeros((len(token_scores)))
Correct_samples_Full_overlap = 
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
  # print(Explanation, token_sorted_top, common_token)
  if common_token/len(Explanation)>.5:
    Correct_samples[i] = 1
  # raise KeyboardInterrupt
print("########## Model:",TheClassifier,"#################")
print('Accuracy percentage:', sum(Correct_samples)/len(token_scores))
print('Number of samples with the ground truth explanations:',sum(Correct_samples),'Number of total samples:', len(token_scores))  