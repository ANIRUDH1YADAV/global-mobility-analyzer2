import pandas as pd

from us_visa.configuration.azure_config import get_engine as build_azure_engine


class AzureSQLDataIngestion:

    def __init__(self, server, database, username, password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password

    def get_engine(self):
        return build_azure_engine(
            server=self.server,
            database=self.database,
            username=self.username,
            password=self.password,
        )

    def fetch_data(self, table_name="EasyVisa"):
        if not str(table_name).replace("_", "").isalnum():
            raise ValueError("Invalid table name. Use alphanumeric characters and underscores only.")

        engine = self.get_engine()
        query = f"SELECT * FROM [{table_name}]"
        df = pd.read_sql(query, engine)
        return df