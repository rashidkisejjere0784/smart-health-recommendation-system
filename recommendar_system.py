import data_cleaning as data_cleaner

DataCleaner = data_cleaner.DataCleaner()

latitude = 344
longitude = 3232
rating = 3.1
payment = "insurance"
care_system = "Public"
services = "I need a dentist"

data_vector, Full_vectors = DataCleaner.get_vector_matrices(services, latitude=latitude,
                                 longitude=longitude, 
                                 rating=rating, payment=payment, care_system=care_system)

print(data_vector.shape)

print(Full_vectors.shape)