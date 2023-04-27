import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import os
import sys 

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import random
import argparse as ap
from datasets import *
#parser = ap.ArgumentParser()
#parser.add_argument("epochs", default=50)
#parser.add_argument("batch_size", default=16)
#arguments = parser.parse_args()
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', padding=True, truncation=True)
model = GPT2LMHeadModel.from_pretrained('gpt2')

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
else:
    device = torch.device("cpu")


#parameters
epochs = 50
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
batch_size = 1
# TODO: add in the train and validation datasets

firstdataset = load_dataset("Cropinky/rap_lyrics_english")
firstdataset.set_format("torch")
#firstdataset = firstdataset['train']
#inputdata = Dataset.from_dict(firstdataset)

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    a = 0 
    for d in tqdm(txt_list['text'], "loading data"):
        #txt = txt_list['train'][i]['text']
        
        encodings_dict = tokenizer('<|startoftext|>'+ d + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

        self.input_ids = (torch.tensor(encodings_dict['input_ids']))
        self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        a += 1
        if a > 2000:
            break
    
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 

print("-------------begin dataset creation------------")
#dataset = GPT2Dataset(firstdataset, tokenizer, max_length=768)
def tokenization(data):
    print(data)
    encodings_dict = tokenizer('<|endoftext|>'+ data + '<|endoftext|>', truncation=True, max_length=768, padding=768)
    return encodings_dict
data = firstdataset['train']
#dataset = data.map(lambda x: tokenization(x['text']), batched=True)
dataset = data.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
# Split into training and validation sets
#train_size = int(0.9 * 2000)
#val_size = 2000 - train_size

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#print('{:>5,} training samples'.format(train_size))
#print('{:>5,} validation samples'.format(val_size)) 

def collate_fn(dataval):
    print(dataval)
    inputs = [torch.tensor(d['input_ids']) for d in dataval]
    #labels = [1 for d in dataval]
    inputs = pad_sequence(dataval, batch_first=True) #(4)
    #labels = inputs
    return {
        'tokenized_input': inputs
        #'label': labels
    }

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            dataset,  # The training samples.
            #sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size, # Trains with this batch size.
            #collate_fn= collate_fn,
            drop_last=True
        )

print(train_dataloader)
# For validation the order doesn't matter, so we'll just read them sequentially.
'''validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            #sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size, # Evaluate with this batch size.
            #collate_fn = collate_fn,
            drop_last=True
        )'''

# this produces sample output every 100 steps
sample_every = 100

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

total_t0 = time.time()

training_stats = []

model = model.to(device)




for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in tqdm(enumerate(train_dataloader), ' batch'):
        #print(batch[''])
        #print(batch['input_ids'].shape())
        b_input_ids = batch['input_ids'].to(device)
        b_labels = batch['input_ids'].to(device)
        b_masks = batch['attention_mask'].to(device)
        print(b_input_ids.size())
        print(torch.max(b_input_ids))
        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % 100 == 0 and not step == 0:
            #torch.save(model.state_dict(), 'rapgenmid.pth')
            torch.save(model, 'rapgenmid.pt')

            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}'.format(step, len(train_dataloader), batch_loss))

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print(sample_output)
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()
            

        loss.backward()

        optimizer.step()

        scheduler.step()
    

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    '''# ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))'''

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            #'Valid. Loss': avg_val_loss
        }
    )

print("")
print("Training complete!")

torch.save(model.state_dict(), 'rapgen.pth')
