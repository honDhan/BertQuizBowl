import pandas as pd
import numpy as np
import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

num_questions = 1000

## ## ### ## ##
# Import Data #
## ## ### ## ##

train = pd.read_json('../nlp-hw/data/qanta.train.json')
# test = pd.read_json('../nlp-hw/data/qanta.test.json')
# dev = pd.read_json('../nlp-hw/data/qanta.dev.json')

train_questions = ['[CLS] ' + q['text'] + ' [SEP]' for q in train['questions']][0:num_questions]
train_answers =  [q['answer'] for q in train['questions']][0:num_questions]

## ### ## ## ### ##
# Setup Tokenizer #
## ### ## ## ### ##

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
tokenized_questions = [tokenizer.tokenize(q) for q in train_questions]

## ## ### ## ##
# Setup Input #
## ## ### ## ##

max_len = 512
input_ids = [tokenizer.convert_tokens_to_ids(q) for q in tokenized_questions]
input_ids = pad_sequences(input_ids, maxlen = max_len, dtype = "long", truncating = "post", padding = "post")

le = LabelEncoder()
label_ids = le.fit_transform(train_answers)

attention_masks = []
for seq in input_ids:
  seq_mask = [float(i > 0) for i in seq]
  attention_masks.append(seq_mask)

## ## ### ## ##
# Setup Input #
## ## ### ## ##

# Can change to the actual files rather than splitting

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, label_ids, random_state = 2018, test_size = 0.1)
train_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, random_state = 2018, test_size = 0.1)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader 
train_data = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = torch.utils.data.RandomSampler(train_data)
train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
validation_sampler = torch.utils.data.SequentialSampler(validation_data)
validation_dataloader = torch.utils.data.DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

## ### ##
# Model #
## ### ##

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_questions)

## ## ### ##
# Training #
## ### ## ##

# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
   {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
   {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr = 2e-5)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
   pred_flat = np.argmax(preds, axis=1).flatten()
   labels_flat = labels.flatten()
   return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
train_loss_set = []
epochs = 4

for i in range(epochs):  
   # Train
   model.train()
   
   tr_loss = 0
   nb_tr_examples, nb_tr_steps = 0, 0

   for step, batch in enumerate(train_dataloader):
      b_input_ids, b_input_mask, b_labels = batch

      optimizer.zero_grad()

      outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
      loss = outputs[0]
      train_loss_set.append(loss.item())

      loss.backward()

      optimizer.step()

      tr_loss += loss.item()
      nb_tr_examples += b_input_ids.size(0)
      nb_tr_steps += 1

   print("Train loss: {}".format(tr_loss/nb_tr_steps))
       
   # Evaluation
   model.eval()

   eval_loss, eval_accuracy = 0, 0
   nb_eval_steps, nb_eval_examples = 0, 0

   for batch in validation_dataloader:
      b_input_ids, b_input_mask, b_labels = batch
      
      with torch.no_grad():
         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask) 
         logits = outputs[0] 

      tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1

   print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))