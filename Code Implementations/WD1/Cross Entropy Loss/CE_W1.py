MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE= 1e-05

targets_settings =[
   ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational','Emotional'],
   ['Physical', 'Intellectual', 'Social', 'Vocational','Spiritual_Emotional'],
   [ 'Intellectual', 'Social', 'Vocational','Physical_Spiritual_Emotional'],
   [ 'Social', 'Intellectual_Vocational','Physical_Spiritual_Emotional']
]
classifier_index = 2 #Models
target_index = 2 # [0: 6-dim, 1:5-dim, 2:4-dim, 3:3-dim]
ran_index = 0 #d[0:200, 1:345, 2:546]

Classifiers = ["nghuyong/ernie-2.0-en", "bert-base-uncased","roberta-base" ,"emilyalsentzer/Bio_ClinicalBERT", "xlnet-base-cased",'nlptown/bert-base-multilingual-uncased-sentiment', "mental/mental-bert-base-uncased"]
Classifiers_Abs = ["ERNIE", "BERT", "RoBERTa", "ClinicalBERT", "XLNET", "PsychBERT", "Mental-BERT"]
TheClassifier = Classifiers[classifier_index]
TheClassifier_Abstract = Classifiers_Abs[classifier_index]

rand_states = [200, 345, 546]


rand_state = rand_states[ran_index]

target_List = targets_settings[target_index]
dimension = len(target_List)

import pandas as pd
import numpy as np

data=pd.read_csv("MultiLabel_WD.csv") # MultiLabel_WD.csv is the first Dataset

data=data.astype({'Spiritual':'float', 'Physical':'float', 'Intellectual':'float', 'Social':'float', 'Vocational':'float',
       'Emotional':'float'})

# print(data)
labels_name = data.columns

labels_name = labels_name[1:]
labels = data[labels_name]

counts = np.zeros(dimension)
for i in range(len(data)):
  for j in range(dimension):
    if data.loc[i][labels_name[j]]>0:
      counts[j] += 1
      
labels_dic ={labels_name[i]:counts[i] for i in range(dimension)}
data_d5 = data.copy(deep=True)
data_d4 = data.copy(deep=True)
data_d3 = data.copy(deep=True)

data_d5['Spiritual_Emotional'] = data_d5[[ 'Spiritual', 'Emotional']].max(axis=1)
data_d5 = data_d5.drop(['Spiritual', 'Emotional'], axis=1)

data_d4['Physical_Spiritual_Emotional'] = data_d4[['Physical', 'Spiritual', 'Emotional']].max(axis=1)
data_d4 = data_d4.drop(['Physical','Spiritual', 'Emotional'], axis=1)

data_d3 = data_d4.copy(deep=True)
data_d3['Intellectual_Vocational'] = data_d3[['Intellectual', 'Vocational']].max(axis=1)
data_d3 = data_d3.drop(['Intellectual', 'Vocational'], axis=1)

if dimension==5:
   data=data_d5
elif dimension==4:
   data=data_d4
elif dimension==3:
   data=data_d3


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(TheClassifier)
import torch

class CustomDataset(torch.utils.data.Dataset):
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
def finalLabels2(predicted_list,val_list):

  indices=np.array(predicted_list).argsort()[::-1][:int(sum(val_list))]
  # argsort()[:-1][:n]
  # print(predicted_list,np.array(predicted_list[i]).argsort()[::-1][:int(sum(val_list[i]))])
  for j in range(len(predicted_list)):
    if j in indices:
      predicted_list[j]=1.0
    else:
      predicted_list[j]=0.0
  return predicted_list


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

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained(TheClassifier, output_hidden_states=True, output_attentions=True, return_dict=True)# BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, dimension)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        # output_dropout = self.dropout(output.pooler_output)
        # output = self.linear(output_dropout)
        # return output
        output_with_attention = output
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output,output_with_attention

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    # return torch.nn.BCEWithLogitsLoss()(outputs[:,:-1], targets)
    return torch.nn.CrossEntropyLoss()(outputs, targets)

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
        # print("shape:",targets.shape)
        # print("shape:",outputs.shape)
        # raise KeyboardInterrupt

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        # tar, outp = loss_fn(outputs, targets.type(torch.int64))
        # return tar, outp

        # print(len(targets))
        # loss2 = loss_fn2(outputs, targets)
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
    #   save_ckp(checkpoint, False, checkpoint_path, best_model_path)

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        # save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model

import shutil, sys

model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, "/ckpt_path", "/best.pt")

#
import random
# for random_indes in range(len(val_df)): #random.randint(0, range(len(val_df)))
random_indes = 166
example  = val_df.loc[random_indes]['text']
target  = [val_df.loc[random_indes][j+1] for j in range(dimension)]
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
    final_output = torch.sigmoid(output)
    # print("Final output:", final_output)
    # print("probabilities vector:", torch.nn.Softmax(output))
    # print("Probailities normalized:", torch.nn.functional.normalize(output, p=1.0, dim = 1))
    print("(",random_indes,")", torch.nn.functional.normalize(final_output, p=1.0, dim = 1))
    print("Target:", target)
    print(example)
    print("***************")
    # print(output_attentions)

    attentions = output_attentions.attentions[11][0][0]
    # Calculate attention scores for each token
    attention_scores = torch.sum(attentions, dim=0)
    # Normalize attention scores
    normalized_scores = attention_scores / torch.sum(attention_scores)

    # Associate each score with its corresponding token
    token_score = {}
    for j in range(len(normalized_scores)):
        token = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))[j]
        token = token.replace('Ġ','')
        token_score[token] = int(255*normalized_scores[j].item())

    import pickle
    # create a binary pickle file 
    f = open("166samples_6dim.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(token_score,f)

    # close file
    f.close()

    # text = ""
    # for token, shade in token_score.items():
    #         print(token,":",shade)
            # shade = int(255 * score)  # calculate shade between 0 and 255
            # color = f"\033[38;2;{shade};0;0m"  # create color with specified shade of red
            # text += f"{color}{token}\033[0m "  # add color to token and append to text
            # # f.write(f'{text}\n\n')
    # print(text)
    # token_scores.append(token_score)
      


    # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
    # final=[0 if i<0.4 else 1 for i in final_output[0]]
    # print("final",final)
    # print(final_output[0][:-1], target)
    # temp = finalLabels2(final_output[0][:-1],target)





final_list=[]
# for i in val_df['text']:
#   example = i
for i in range(len(val_df)):
  example  = val_df.loc[i]['text']
  target  = [val_df.loc[i][j+1] for j in range(dimension)]
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
      output,_ = model(input_ids, attention_mask, token_type_ids)
      temp = torch.Tensor(target).type(torch.int64).to(device)
      loss, reservation = loss_fn(output,temp.reshape([1,dimension]))
      final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
      # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
      # final=[0 if i<0.4 else 1 for i in final_output[0]]
      # print("final",final)
      # print(final_output[0][:-1], target)
      temp = finalLabels2(final_output[0][:-1],target)
    #   print(temp)
    #   print(len(temp))
    #   print(target)
    #   print(len(target))

      final_list.append(temp+target+torch.Tensor.tolist(reservation))

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