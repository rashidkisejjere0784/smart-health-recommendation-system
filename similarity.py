import numpy as np
import pandas as pd
from numpy.linalg import norm
import joblib

services_vectors = joblib.load('services vectors.pkl')


def calculate_cosine_similarity_grounded(point,Full_encoded_data, n=10):
  cosine = np.dot(Full_encoded_data, point)/(norm(Full_encoded_data, axis=1)*norm(point))
  top_choices = np.argsort(cosine)[-n:]
  return top_choices

def get_grounded_predictions(point, full_data, n = 10, services_index = 205):
  point_services = point[-services_index:]
  top_choices = calculate_cosine_similarity_grounded(point_services, services_vectors, n= n)

  To_consider_vectors = full_data[top_choices, :-services_index]
  other_point_features = point[:-services_index]

  choices = calculate_cosine_similarity_grounded(other_point_features, To_consider_vectors, n = n)
  return top_choices[choices]