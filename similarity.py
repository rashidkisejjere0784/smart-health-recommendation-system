import numpy as np
import pandas as pd
from numpy.linalg import norm


def calculate_cosine_similarity(point, Full_Encoded_data, n=3):
  cosine = np.dot(Full_Encoded_data, point)/(norm(Full_Encoded_data, axis=1)*norm(point))
  top_choices = np.argsort(cosine)[-n:]
  return top_choices