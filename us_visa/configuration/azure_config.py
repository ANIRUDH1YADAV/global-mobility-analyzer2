import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def _get_sql_server_driver() -> str:
    """Return the best available SQL Server ODBC driver on this machine."""
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError(
            "pyodbc is not installed. Install it with: pip install pyodbc"
        ) from exc

    installed_drivers = {driver.strip() for driver in pyodbc.drivers()}
    preferred_order = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server",
    ]

    for driver in preferred_order:
        if driver in installed_drivers:
            return driver

    raise RuntimeError(
        "No SQL Server ODBC driver found. Install 'ODBC Driver 18 for SQL Server'."
    )


def get_engine(
    server: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    driver: str | None = None,
):
    """Create a SQLAlchemy engine for Azure SQL using environment defaults."""
    server = server or os.getenv("AZURE_SERVER")
    database = database or os.getenv("AZURE_DB")
    username = username or os.getenv("AZURE_USER")
    password = password or os.getenv("AZURE_PASSWORD")
    driver = driver or os.getenv("AZURE_ODBC_DRIVER") or _get_sql_server_driver()

    missing = [
        key
        for key, value in {
            "AZURE_SERVER": server,
            "AZURE_DB": database,
            "AZURE_USER": username,
            "AZURE_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    params = urllib.parse.quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True,
    )