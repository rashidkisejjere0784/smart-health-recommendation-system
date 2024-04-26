import pandas as pd
import re
import joblib
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#load the vectorizer
vectorizer = joblib.load("./services vectorizer.pkl")
cat_data_encoder = joblib.load("./care_system and payment_encoder encoder.pkl")
services_vectors = joblib.load("services vectors.pkl")

num_df = pd.read_csv("num cols.csv")

stopwords = stopwords.words()

class DataCleaner():
    def __init__(self):
        self.vectorizer = vectorizer
        self.cat_data_encoder = cat_data_encoder
        self.lemmatizer = WordNetLemmatizer().lemmatize
        self.numerical_df = num_df

    def clean_text(self, txt: str) -> str:
        txt = " ".join([self.lemmatizer(word) for word in txt.split()])
        txt = re.sub('sergeries', 'sergery', txt)
        txt = re.sub('surgical', 'sergery', txt)
        txt = re.sub('department', '', txt)
        txt = re.sub(r'\(.+\)', '', txt)
        txt = re.sub(r'[-\.]', '', txt)
        txt = re.sub('services', '', txt)
        txt = re.sub('service', '', txt)
        txt = re.sub('care', '', txt)
        txt = re.sub('unit', '', txt)
        txt = re.sub('tb', 'tuberculosis', txt)
        return txt
    
    def get_vetors(self, txt : str) -> str:
        txt = self.clean_text(txt)
        vectors = self.vectorizer.transform(txt)
        return vectors
    

    def get_vector_matrices(self, services : str, latitude = None, longitude = None, rating = None, care_system = None, payment = None) -> list:
        gen_vector = np.array([])
        Full_encoded_data = np.array([])
        services = self.clean_text(services.lower())

        if latitude is not None:
            gen_vector = np.array([latitude])
            Full_encoded_data = num_df.iloc[:, 1].values.reshape(-1, 1)
        
        if longitude is not None:
            if len(gen_vector) == 0:
                gen_vector = np.array([longitude])
                Full_encoded_data = num_df.iloc[:, 2].values.reshape(-1, 1)

            else:
                gen_vector = np.concatenate([[longitude], gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df.iloc[:, 2].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if rating is not None:
            if len(gen_vector) == 0:
                gen_vector = np.array([rating])
                Full_encoded_data = num_df.iloc[:, 3].values.reshape(-1, 1)

            else:
                gen_vector = np.concatenate([[rating], gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df.iloc[:, 3].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if (care_system is not None) and (payment is not None):
            cat_encoded = self.cat_data_encoder.transform([[care_system, payment]])[0]
            if len(gen_vector) == 0:
                gen_vector = cat_encoded
                Full_encoded_data = num_df.iloc[:, 4:]
            else:
                gen_vector = np.concatenate([cat_encoded, gen_vector], axis = 0)

                Full_encoded_data = np.concatenate([num_df.iloc[:, 4:].values, Full_encoded_data], axis = 1)

        service_vector = self.vectorizer.transform([services])[0].toarray()
        gen_vector = np.concatenate([service_vector.ravel(), gen_vector])
        Full_encoded_data = np.concatenate([services_vectors, Full_encoded_data], axis = 1)

        return gen_vector, Full_encoded_data


