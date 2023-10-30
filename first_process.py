import os
import pandas as pd
path = r'.\10km'
# 处理数据部分：将 2018/01/01 00:00:00 格式的date分为5列
for file in os.listdir(path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, file))
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['doy'] = df['date'].dt.dayofyear
        df = df[['year', 'month', 'day', 'hour', 'doy', 't2m', 'sp', 'rh', 'tp', 'flash']]
        df.dropna(inplace=True)
        df.to_csv(os.path.join(path, file), index=False)