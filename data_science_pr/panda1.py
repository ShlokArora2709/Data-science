import pandas as pd


data = {
    'City': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
    'Atemp': [30, 25, 27, 29, 28],
    'Ahumid': [80, 60, 75, 85, 70]
}


data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
    'Atemp': [13, 19, 11, 20, 25],
    'Ahimid': [65, 70, 55, 70, 75]
}

usdf = pd.DataFrame(data)

inddf = pd.DataFrame(data)

ndf=pd.concat([usdf,inddf],keys=["us","india"])
print(ndf.loc["us"])
