import duckdb
import data_collector as dc
import pandas as pd
import cleaning
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select, table, literal_column, MetaData, Table

def load_to_db(df, table_name):
    conn = duckdb.connect("database.duckdb")

    conn.register("temp_df", df)

    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM temp_df""")

    conn.close()

sources = dc.load_sources()
for name, url in sources.items():
    raw_df = dc.get_latest_as_df(name)
    cleaned_df = cleaning.clean_data(raw_df)
    raw_name = f"raw_{name}"
    cleaned_name = f"cleaned_{name}"

    if raw_df is not None:
        load_to_db(raw_df, raw_name)
    if cleaned_df is not None:
        load_to_db(cleaned_df, cleaned_name)


def get_from_db(table_name: str) -> pd.DataFrame:
    conn = duckdb.connect("database.duckdb")
    try:
        result = conn.table(table_name).to_df()
        return result
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None
    finally:
        conn.close()

# def get_from_db(table_name: str) -> pd.DataFrame:
#     try:
#         db_engine = create_engine('duckdb:///./database.duckdb')
#         # Tiszta SQLAlchemy kifejezésfa objektum:
#         # Létrehoz egy TableClause objektumot a dinamikus névből
#         tbl = table(table_name)

#         metadata = MetaData()
#         tbl = Table(table_name, metadata, autoload_with=db_engine)
        
#         # SQLAlchemy generálja le a megfelelő SQL-t, escaped azonosítókkal
        
#         with Session(db_engine) as session:
#             stmt = select(tbl)
#             result = session.execute(stmt)
#             df = pd.DataFrame(result.fetchall(), columns=result.keys())
#             return df
#     except Exception as e:
#         print(f"Error fetching data from {table_name}: {e}")
#         return None
    
df = get_from_db("gold_co2_emissions")
print("Data from gold_co2_emissions:", df)

#Jelenleg SQLAlchemy Coret használtam és fogok a fenti fv-ben hogy ne legyen string benne. Ez az ORM-et így nem használja. Kérdés legyen-e benne
#ORM és ha igen akkor hogyan? 