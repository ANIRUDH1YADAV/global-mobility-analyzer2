from us_visa.components.azure_data_ingestion import AzureSQLDataIngestion

config = {
    "server": "globalmobility.database.windows.net",
    "database": "GlobalMobilityAnalyzer",
    "username": "CloudSab50becab",
    "password": "AN1rudh42",
}

ingestion = AzureSQLDataIngestion(**config)

df = ingestion.fetch_data("EasyVisa")

print("✅ Ingestion working")
print(df.shape)