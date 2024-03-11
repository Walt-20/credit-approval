import pandas as pd

df = pd.read_csv('application_record.csv')
df = df[['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']]
df['AMT_INCOME_TOTAL'] = (df['AMT_INCOME_TOTAL'] > 50000).astype(int)
df['DAYS_EMPLOYED'] = (df['DAYS_EMPLOYED'] > 90).astype(int)

df.to_csv('labels.csv', index=False)