import data_collector as dc
import pandas as pd

dfs = []
for name  in dc.load_sources().keys():
    print(f"Loading {name} dataset...")
    df = dc.get_latest_as_df(name)
    print(df)
    dfs.append(df)

