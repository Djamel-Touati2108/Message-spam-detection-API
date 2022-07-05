import pickle
import re
from keras.preprocessing.sequence import pad_sequences
import contractions
from nltk.stem import WordNetLemmatizer
import nltk
import app


def spam_detector(sms):
  
  loaded_model=app.loaded_model

  prepared_text=prepare_data(sms)
  result = loaded_model.predict(prepared_text).item()*100
  
  return result


  
  
  


def prepare_data(sms):
  with open('TOKENIZER', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

  lem = WordNetLemmatizer()
  max_length_sequence=79
  stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

  sms = contractions.fix(sms) # converting shortened words to original (Eg:"I'm" to "I am")
  sms = sms.lower() # lower casing the sms
  sms = re.sub(r'£|\$', "money-symbol", sms).strip() #replacing money symbols
  sms = re.sub("[^a-z ]", "", sms) # removing symbols and numbers
  sms = re.sub(r'http\S+', ' webaddress ', sms) #replacing urls
  sms = sms.split() #splitting
  # lemmatization and stopword removal
  sms = [lem.lemmatize(word) for word in sms if not word in stop_words]
  sms = " ".join(sms)
  textDataArray = [sms]
  text_to_sequence = loaded_tokenizer.texts_to_sequences(textDataArray)
  text = pad_sequences(text_to_sequence, maxlen=max_length_sequence, padding = "pre") 
  return text