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
cat_data_encoder_dict = joblib.load("./cat_data_encoder_dict.pkl")
services_vectors = joblib.load("services vectors.pkl")
location_vectorizer = joblib.load("location vectorizer.pkl")
location_vectors = joblib.load("location vectors.pkl")

num_df = pd.read_csv("num cols.csv")

stopwords = stopwords.words()

class DataCleaner():
    def __init__(self):
        self.vectorizer = vectorizer
        self.cat_data_encoder_dict = cat_data_encoder_dict
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
    

    def get_vector_matrices(self, services : str, Location=None, Monday=None,Tuesday=None, 
                            Wednesday=None, Thursday=None, Friday=None, Saturday=None,
                              Sunday=None, rating = None, care_system = None, payment = None) -> list:
        gen_vector = np.array([])
        Full_encoded_data = np.array([])
        services = self.clean_text(services.lower())

        if Location is not None:
            gen_vector = np.array(location_vectorizer.transform([Location]).toarray())[0]
            Full_encoded_data = np.array(location_vectors)
        
        if care_system is not None:
            care_system = cat_data_encoder_dict['Care system'].transform([care_system])
            if len(gen_vector) == 0:
                gen_vector = care_system
                Full_encoded_data = num_df['Care system'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([care_system, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Care system'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if payment is not None:
            payment = cat_data_encoder_dict['payment'].transform([payment])
            if len(gen_vector) == 0:
                gen_vector = payment
                Full_encoded_data = num_df['payment'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([payment, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['payment'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if Monday is not None:
            Monday = cat_data_encoder_dict['Monday'].transform([Monday])
            if len(gen_vector) == 0:
                gen_vector = Monday
                Full_encoded_data = num_df['Monday'].values.reshape(-1, 1)

            else:
                gen_vector = np.concatenate([Monday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Monday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)
            

        if Tuesday is not None:
            Tuesday = cat_data_encoder_dict['Tuesday'].transform([Tuesday])
            if len(gen_vector) == 0:
                gen_vector = Tuesday
                Full_encoded_data = num_df['Tuesday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Tuesday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Tuesday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)
        
        if Wednesday is not None:
            Wednesday = cat_data_encoder_dict['Wednesday'].transform([Wednesday])
            if len(gen_vector) == 0:
                gen_vector = Wednesday
                Full_encoded_data = num_df['Wednesday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Wednesday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Wednesday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)
        
        if Thursday is not None:
            Thursday = cat_data_encoder_dict['Thursday'].transform([Thursday])
            if len(gen_vector) == 0:
                gen_vector = Thursday
                Full_encoded_data = num_df['Thursday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Thursday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Thursday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if Friday is not None:
            Friday = cat_data_encoder_dict['Friday'].transform([Friday])
            if len(gen_vector) == 0:
                gen_vector = Friday
                Full_encoded_data = num_df['Friday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Friday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Friday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if Saturday is not None:
            Saturday = cat_data_encoder_dict['Saturday'].transform([Saturday])
            if len(gen_vector) == 0:
                gen_vector = Saturday
                Full_encoded_data = num_df['Saturday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Saturday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Saturday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if Sunday is not None:
            Sunday = cat_data_encoder_dict['Sunday'].transform([Sunday])
            if len(gen_vector) == 0:
                gen_vector = Sunday
                Full_encoded_data = num_df['Sunday'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([Sunday, gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['Sunday'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        if rating is not None:
            if len(gen_vector) == 0:
                gen_vector = [rating]
                Full_encoded_data = num_df['rating'].values.reshape(-1, 1)
            
            else:
                gen_vector = np.concatenate([[rating], gen_vector], axis = 0)
                Full_encoded_data = np.concatenate([num_df['rating'].values.reshape(-1, 1), Full_encoded_data], axis = 1)

        
        service_vector = self.vectorizer.transform([services])[0].toarray()
        gen_vector = np.concatenate([gen_vector,service_vector.ravel()])

        print(gen_vector.shape, service_vector.shape, Full_encoded_data.shape, services_vectors.shape)
        Full_encoded_data = np.concatenate([Full_encoded_data, services_vectors], axis = 1)

        return gen_vector, Full_encoded_data
