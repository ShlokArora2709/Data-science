import pandas as pd
from typing import List

def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    return pd.DataFrame(columns=["student_id", "age"],data=student_data)



def getDataframeSize(players: pd.DataFrame) -> List[int]:
    return list(players.shape)
    

def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.head(3)


def selectData(students: pd.DataFrame) -> pd.DataFrame:
    return students[students['student_id']==101][['name','age']]
    
def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus']=employees['salary']*2
    return employees