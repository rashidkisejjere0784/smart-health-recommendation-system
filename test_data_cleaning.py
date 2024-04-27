import data_cleaning as DataCleaning
import similarity
import pandas as pd

dataClean  = DataCleaning.DataCleaner()
data = pd.read_excel("Hospital Data.xlsx")

Location = "Kampala"
Monday = "Open 24 hours"
Tuesday = "Open 24 hours"
Wednesday = "Open 24 hours"
Thursday = "Open 24 hours"
Friday = "Open 24 hours"
Saturday = "Open 24 hours"
Sunday = "Open 24 hours"
service = "I have cancer"
payment = "cash"
care_system = "Public"
rating = 4

point_vector, full_data = dataClean.get_vector_matrices(services=service,
                                                         Location=Location,rating=rating,
                                                         care_system=care_system)

print(point_vector.shape, full_data.shape)

hospital_indicies = similarity.get_grounded_predictions(point_vector, full_data)

print(data['Hospital'].iloc[hospital_indicies])