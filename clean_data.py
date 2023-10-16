import os 
import pandas as pd
import nltk
import re
import string

def import_data(hard_coded_file=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data')
    files_in_data = os.listdir(data_dir)

    while True:
        if hard_coded_file == None:
            print('Please choose a file that you want to use:')
            for i, file in enumerate(files_in_data):
                print(f'{i+1}. {file}')
        
            user_input = int(input('> ')) - 1
        
            if 0 <= user_input < len(files_in_data):
                print(files_in_data[user_input])
                file_path = os.path.join(data_dir, files_in_data[user_input])
                return pd.read_csv(file_path)
            else:
                print('Invalid option. Please select one of the provided options.')
        else:
            print('File name is present')
            file_path = os.path.join(data_dir, hard_coded_file)
            return pd.read_csv(file_path)    
            
def clean_data(data):
    # Lowercase the data
    data['content'] = data['content'].str.lower()
    
    # Remove the stopwords
    data['content'] = data['content'].apply(remove_usernames)

    # Deduplicate the data 
    index = data[data['content'].duplicated() == True].index
    data.drop(index, axis=0, inplace=True)
    data.reset_index(inplace=True, drop=True)

    # Remove punctuation
    data['content'] = data['content'].apply(remove_punctuation)

    new_data = data[data['sentiment'].isin(['happiness','sadness','anger'])]

    new_data.reset_index(drop=True, inplace=True)

    return new_data
    
def lower_case_words(text):
    #text = [t.lower() for t in text]
    #text = text.lower()
    words = text.split()
    text = " ".join([word.lower() for word in words])
    return text

def remove_punctuation(text):
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)

def remove_usernames(text):
    pattern = r'@\w+'

    cleaned_text = re.sub(pattern,'',text)
    return cleaned_text

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_usernames(sentence)
    sentence = remove_punctuation(sentence)

    return sentence