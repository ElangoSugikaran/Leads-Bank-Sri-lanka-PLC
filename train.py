#Building Training pipeline - creating actual training data - application of preprocessing techniques ( in nltk_utils.py)
#Load jason file
import json 
from nltk_utils import tokenize, stem, bag_of_words #importinf utility part from nltk_ytils.py
import numpy as np #covert lables to numpy array

#Import what we need for pytorch to Create pytorch dataset from training data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = [] #Empty list to collect all the different patterns
tags = [] #Empty list to collect all the different tags
xy = [] #Empty list which will later holds all our patterns and tags
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
     # add to tag list
    tags.append(tag)
     #Loop over all the different patterns
    for pattern in intent ['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list in to all words array
        all_words.extend(w)
         # add to xy pair
        xy.append((w, tag)) #collect pattern and currosponding tag by this tuple

 
# stem and lower each word - exclude punctuation characters       
ignore_words =['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]#apply stemming
# remove duplicates and sort
all_words = sorted(set(all_words))#sorting words + creating 'set' to remove duplicate elements
tags = sorted(set(tags))
#print(tags)

# create training data - creating bags of words             
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
      # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
       # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)#Because of CrossEntropyLoss don't have to care about 1 hot ncoding here, only want to have class lables in this pattern

 
#Create pytorch dataset from training data     
x_train = np.array(x_train)
y_train = np.array(y_train)

class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
      # support indexing such that dataset[i] can be used to get i-th sample -- dataset[idx] --> later we can access data with index  
    #dataset[idx]    
    def __getitem__(self, index):
        return self.x_data(index), self.y_data(index)
    
    def __len__(self):
        return self.n_samples
    
#Define Hyper-parameters 
batch_size = 8 
hidden_size = 8 
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000
#print(input_size, len(all_words))
#print(output_size, tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model - actual training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:    #use our training loader + unpacking (words, lables)
        words = words.to(device)    #puch pack to the device
        labels = labels.to(dtype=torch.long).to(device) #puch lables to the device
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

#save data - creating a dictionary
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


