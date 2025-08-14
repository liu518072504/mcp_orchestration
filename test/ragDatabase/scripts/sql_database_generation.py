import sqlite3
import pandas as pd
import os
import argparse

print("Starting script...")

# Argument parser for base and output paths
parser = argparse.ArgumentParser(description="Load all CSVs in base_path into a SQLite database.")
parser.add_argument("--base_path", required=True, help="Path to the folder containing CSV files")
parser.add_argument("--output_path", required=True, help="Path to save the SQLite DB file")
parser.add_argument("--output_db", default="supply_chain.db", help="SQLite database filename (default: supply_chain.db)")
args = parser.parse_args()

# Full path for the SQLite database file
db_full_path = os.path.join(args.output_path, args.output_db)

# Create output directory if it doesn't exist
os.makedirs(args.output_path, exist_ok=True)

# Connect to SQLite database (or create it)
conn = sqlite3.connect(db_full_path)

# List all CSV files in base_path
csv_files = [f for f in os.listdir(args.base_path) if f.endswith(".csv")]

# Loop through CSVs and load each into the database
for csv_file in csv_files:
    table_name = os.path.splitext(csv_file)[0]  # Use filename without extension as table name
    file_path = os.path.join(args.base_path, csv_file)

    try:
        df = pd.read_csv(file_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✅ Loaded table: {table_name}")
    except Exception as e:
        print(f"❌ Failed to load {csv_file}: {e}")

# Close connection
conn.close()
print(f"\n🎉 Database created successfully at: {db_full_path}")
