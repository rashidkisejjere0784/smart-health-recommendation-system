import intellikit as ik
import pandas as pd
import numpy as np

hospital_data = pd.read_excel("Hospital Data.xlsx")

services = []
for i, service in enumerate(hospital_data['cleaned services']):
    services.append(f"{i} -- {service}")

def get_recommendation(
        service : str, Location=None, Monday=None,Tuesday=None, 
                            Wednesday=None, Thursday=None, Friday=None, Saturday=None,
                              Sunday=None, rating = None, care_system = None, payment = None):
    
    top_4 = ik.vector_space_model(service, services, k = 4)

    top_choices = []
    for doc, _ in top_4:
        index = int(doc.split('--')[0].strip())
        top_choices.append(hospital_data.loc[index])

    scores = []
    for hospital in top_choices:
        score = 0
        if Location is not None:
            if str(hospital['Location']).find(Location) != -1:
                score += 3

        if care_system is not None:
            if str(hospital['Care system']).find(care_system) != -1:
                score += 2
        
        if payment is not None:
            if str(hospital['payment']).find(payment) != -1:
                score += 2

        if Monday is not None:
            if str(hospital['Monday']).find(Monday) != -1:
                score += 0.5
        
        if Tuesday is not None:
            if str(hospital['Tuesday']).find(Tuesday) != -1:
                score += 0.5

        if Wednesday is not None:
            if str(hospital['Wednesday']).find(Wednesday) != -1:
                score += 0.5

        if Thursday is not None:
            if str(hospital['Thursday']).find(Thursday) != -1:
                score += 0.5

        if Friday is not None:
            if str(hospital['Friday']).find(Friday) != -1:
                score += 0.5

        if Saturday is not None:
            if str(hospital['Saturday']).find(Saturday) != -1:
                score += 0.5

        if Sunday is not None:
            if str(hospital['Sunday']).find(Sunday) != -1:
                score += 0.5

        if (rating is not None) and (rating != 0):
                if hospital['rating'] is not np.nan:
                    score += hospital['rating']

        scores.append(score)

    scores = np.argsort(scores)
    top_choices = [top_choices[i] for i in scores[::-1]]
    print(top_choices)

    return top_choices