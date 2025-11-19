import pandas as pd

def parse_data():
    df0 = pd.read_csv('student-mat.csv', sep = ';')
    df = df0[  ['studytime', 'absences', 'health', 'Medu', 'Fedu', 'G3']  ]
    df = df.rename( columns={'Medu':'medu', 'Fedu':'fedu', 'G3':'grade'} )
    return df

