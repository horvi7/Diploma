import data_collector as dc
import pandas as pd

def load_and_clean_data() -> list[pd.DataFrame]:
    dfs = []
    for name in dc.load_sources().keys():
        print(f"Loading {name} dataset...")
        df = dc.get_latest_as_df(name)

        if df is None:
            continue
        
        date_col = [date_col for date_col in df.columns if "date" in date_col.lower()]
        for col in date_col:
            df[col] = pd.to_datetime(df[col], errors='raise')
            
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        dfs.append(df)
    return dfs


cleaned_dfs = load_and_clean_data()
