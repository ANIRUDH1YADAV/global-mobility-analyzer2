import pandas as pd
from sqlalchemy import text

from us_visa.configuration.azure_config import get_engine


def main() -> None:
    try:
        engine = get_engine()
        query = text("SELECT TOP 5 * FROM EasyVisa")

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print("Connection successful")
        print(df.head())
    except Exception as exc:
        print("Connection failed")
        print(exc)
        print(
            "If this is IM002, install 'ODBC Driver 18 for SQL Server' and "
            "set AZURE_ODBC_DRIVER if needed."
        )
        raise


if __name__ == "__main__":
    main()