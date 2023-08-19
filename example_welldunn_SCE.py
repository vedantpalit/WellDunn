import pickle
import torch
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE= 1e-05
target =['Physical Aspect', 'Intellectual or Vocational Aspect', 'Social Aspect', 'Spiritual or Emotional Aspect']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = model = AutoModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True, return_dict=True)# BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_with_attention = output
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output, output_with_attention


model_path = "MultiWD_model_RoBERTa_SCE_6_Dimension"
# Input example 1: Once it happens there is no going back and you are stuck in an empty void of darkness forever
# Input example 2: I feel like there are only three things that you absolutely cannot be, dumb, ugly, and useless
text = input("Enter the text for inference: ")

# Load the BERT model
model=BERTClass()
model.to(device)
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# print("model",model)
# raise KeyboardInterrupt
model.load_state_dict(torch.load("ExplainWD_model_RoBERTa_SCE_4_Dimension"))
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# model.eval()

# Tokenize and process the input text
encodings = tokenizer.encode_plus(
    text,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,  # You need to define MAX_LEN
    padding='max_length',
    return_token_type_ids=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)


model.eval()
final_list = []
# print("sdsad",model.forward())
with torch.no_grad():
    input_ids = encodings['input_ids'].to(device, dtype=torch.long)
    attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
    token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
    output, reservation = model(input_ids, attention_mask, token_type_ids)
    final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
    label = target[final_output[0].index(max(final_output[0][:-1]))]
    print("The final predicted label is:",label) 
