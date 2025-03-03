import polars as pl
import re
from glob import glob

# Mapping from house number to washing machine column index (0-based)
washing_machine_mapping = {
    1: 7,  # 5.Washing Machine
    2: 4,  # 2.Washing Machine
    3: 8,  # 6.Washing Machine
    # House 4 is skipped
    5: 5,  # 3.Washing Machine
    6: 4,  # 2.Washing Machine
    7: 7,  # 5.Washing Machine
    8: 6,  # 4.Washing Machine
    9: 5,  # 3.Washing Machine
    10: 7,  # 5.Washing Machine
    11: 5,  # 3.Washing Machine
    # House 12 doesn't have a washing machine listed
    13: 5,  # 3.Washing Machine
    15: 5,  # 3.Washing Machine
    16: 7,  # 5.Washing Machine
    17: 6,  # 4.Washing Machine
    18: 7,  # 5.Washing Machine
    19: 4,  # 2.Washing Machine
    20: 6,  # 4.Washing Machine
    21: 5,  # 3.Washing Machine
}


def process_house_file(file_path):
    # Extract house number from file name
    house_number = int(re.search(r'House_(\d+)', file_path.split('/')[-1]).group(1))

    # Skip House 4 and House 12
    if house_number in [4, 12]:
        return None

    # Get washing machine column index
    washing_machine_idx = washing_machine_mapping.get(house_number)

    if washing_machine_idx is None:
        return None

    # Generate all column names based on the CSV format described in readme
    col_names = ['DateTime', 'UnixTimestamp', 'Aggregate']
    for i in range(1, 10):
        col_names.append(f'Appliance{i}')

    # Skip the header row and read the data
    df = pl.scan_csv(
        file_path,
        skip_rows=1,  # Skip the first row that contains header descriptions
        has_header=False,
        new_columns=col_names,
    )

    # Select only the columns we need and rename washing machine column
    washing_machine_col = col_names[washing_machine_idx]
    df = df.select([
        pl.col('DateTime'),
        pl.col('Aggregate'),
        pl.col(washing_machine_col).alias('washing_machine')
    ])

    # Add house number as a column
    df = df.with_columns(pl.lit(house_number).alias("house_number"))

    # Cast columns to appropriate types
    df = df.with_columns([
        pl.col("DateTime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        pl.col("Aggregate").cast(pl.Float64),
        pl.col("washing_machine").cast(pl.Boolean).cast(pl.Float64) # interested only on binary class for now
    ])

    return df.collect()


# Get all house files
house_files = sorted(glob("data/House_*.csv"))

# Process each house file and collect the results
all_dfs = []
for file in house_files:
    print(f"Processing {file}...")
    df = process_house_file(file)
    if df is not None:
        all_dfs.append(df)

# Concatenate all dataframes
final_df = pl.concat(all_dfs)

# Re-order columns to match the requested output
final_df = final_df.select(
    ["DateTime", "Aggregate", "house_number", "washing_machine"]).rename(
    {"DateTime": "Datetime"})

# Save the result
final_df.write_parquet("washing_machine_data.parquet")

print("Extraction completed. Results saved to washing_machine_data.parquet")