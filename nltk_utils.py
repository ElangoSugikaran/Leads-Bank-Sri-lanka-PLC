#implementing our preprocessing techniques (tokenization, lowering, stemming, bag of words) by using the framework NLTK which is a python library to work with human language data
import nltk
import numpy as np #import numpy for creating bag
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer # nltk.download('punkt') A package from NLTK which is a pretrained tokenizer
stemmer = PorterStemmer()

#creating method for tokenization
def tokenize(sentence):  #split sentence into array of words/tokens(a token can be a word or punctuation character, or number)
  return nltk.word_tokenize(sentence)

#creating method for stemming
def stem(word):
   return  stemmer.stem(word.lower())
 
#creating method for bag of words
def bag_of_words(tokenized_sentence, all_words):
    
    tokenized_sentence = [stem(w) for w in tokenized_sentence] #apply stemming for the tokanize sentence

    bag = np.zeros(len(all_words), dtype=np.float32)  #crating bag and initializing with 0 for each word in the list
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag
  
#checking tokenization
#eg:a="How can I know about your bank?"
#print(a)
#a = tokenize(a)
#print(a)

#checking stem - stemming = find the root form of the word
#rg:words = ["Organize","organizes","organizing"]
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words) --> ["organ", "organ", "organ"]

#return bag of words array:(1 for each known word that exists in the sentence, 0 otherwise)
#sentence = ["hello","hi","are","you"]
#words = ["hi","hello","I","you","thank","cool"]
#Sbog = bag_of_words(sentence, words)
#print(bog)
#bog = [0, 1, 0, 1, 0, 0, 0]