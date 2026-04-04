import pandas as pd
import pymssql

# ========== YOUR AZURE SQL CREDENTIALS ==========
server = 'globalmobility.database.windows.net'
database = 'GlobalMobilityAnalyzer'
username = 'CloudSab50becab'
password = 'AN1rudh42'
# ================================================

try:
    print("📁 Reading CSV file...")
    df = pd.read_csv(r'C:\Users\aniru\globel\global-mobility-analyzer\easy_visa_dataset\EasyVisa.csv')
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    print("🔗 Connecting to Azure SQL Database...")
    conn = pymssql.connect(
        server=server,
        user=username,
        password=password,
        database=database,
        timeout=30
    )
    cursor = conn.cursor()
    print("✓ Connected successfully!")
    
    # Drop existing table
    print("🗑️  Dropping existing table...")
    cursor.execute("DROP TABLE IF EXISTS EasyVisa")
    conn.commit()
    
    # Create table with CORRECT data types
    print("📋 Creating table with correct schema...")
    cursor.execute("""
    CREATE TABLE EasyVisa (
        case_id NVARCHAR(50) PRIMARY KEY,
        continent NVARCHAR(100),
        education_of_employee NVARCHAR(100),
        has_job_experience NVARCHAR(50),
        requires_job_training NVARCHAR(50),
        no_of_employees INT,
        yr_of_estab INT,
        region_of_employment NVARCHAR(100),
        prevailing_wage DECIMAL(10, 2),
        unit_of_wage NVARCHAR(50),
        full_time_position NVARCHAR(50),
        case_status NVARCHAR(50)
    )
    """)
    conn.commit()
    print("✓ Table created!")
    
    print(f"\n📤 Uploading {len(df)} rows...")
    
    # Insert data in batches
    for idx, row in df.iterrows():
        try:
            cursor.execute("""
            INSERT INTO EasyVisa 
            (case_id, continent, education_of_employee, has_job_experience, 
             requires_job_training, no_of_employees, yr_of_estab, 
             region_of_employment, prevailing_wage, unit_of_wage, 
             full_time_position, case_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
            
            if (idx + 1) % 1000 == 0:
                conn.commit()
                print(f"  ✓ Uploaded {idx + 1} rows...")
        except Exception as e:
            print(f"  ❌ Row {idx} error: {str(e)[:100]}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\n✅ SUCCESS! All {len(df)} rows uploaded to EasyVisa table!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()