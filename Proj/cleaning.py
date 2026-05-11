import data_collector as dc
import pandas as pd
import great_expectations as gx

def clean_data(df: pd.DataFrame):
    df = df.copy()

    df.columns = [col.strip().lower().replace("(", " ").replace(")", " ") for col in df.columns]

    df.columns = df.columns.str.lower()
    date_col = [date_col for date_col in df.columns if "date" in date_col]

    for col in date_col:
        df[col] = pd.to_datetime(df[col], errors='raise')
    
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    validate_data(df)
    
    return df



def validate_data(df: pd.DataFrame):

    numeric = [
    "co2",
    "gdp",
    "population",
    "energy_per_capita",
    "energy_per_gdp",
    "primary_energy_consumption"
    ]

    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name="co2_data_source")
    data_asset = data_source.add_dataframe_asset(name="co2_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("co2_batch")
    batch_parameters = {"dataframe": df}

    batch_definition = (context.data_sources.get("co2_data_source").get_asset("co2_data_asset").get_batch_definition("co2_batch"))

    expectations = []
    
    for col in ["name", *numeric]:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    for col in numeric:
        expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0, max_value=None))

    expectations.append(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["name", "year"]))

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)
    
    suite = gx.ExpectationSuite(name="co2_validation_suite", expectations=expectations)

    context.suites.add_or_update(suite)

    validation_definition = gx.ValidationDefinition(data=batch_definition, suite=suite, name="co2_validation_definition")

    context.validation_definitions.add_or_update(validation_definition)

    checkpoint = gx.Checkpoint(name="co2_checkpoint", validation_definitions=[validation_definition],
        actions=[
            gx.checkpoint.actions.UpdateDataDocsAction(
                name="update_data_docs"
            )
        ],
        result_format={"result_format": "SUMMARY"}
    )

    context.checkpoints.add_or_update(checkpoint)

    results = checkpoint.run(
        batch_parameters={"dataframe": df}
    )

    context.open_data_docs()