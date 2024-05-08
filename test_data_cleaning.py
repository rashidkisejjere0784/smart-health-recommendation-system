import pandas as pd
import get_recommendation


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

recomedentation = get_recommendation.get_recommendation(service, Location, 
                                                        Monday, Tuesday, Wednesday,
                                                          Thursday, Friday, Saturday, Sunday, 
                                                          rating, care_system, payment)

print(recomedentation)
