import data_collector as dc
import pandas as pd

def clean_data(df: pd.DataFrame):
    df = df.copy()

    df.columns = [col.strip().lower().replace("(", " ").replace(")", " ") for col in df.columns]

    date_col = [date_col for date_col in df.columns if "date" in date_col.lower()]
    for col in date_col:
        df[col] = pd.to_datetime(df[col], errors='raise')
    
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    return df
