import pandas as pd

# Input and output CSV paths
input_csv  = "500mol.csv"
output_csv = "500mol_fixed.csv"

# Try reading the CSV
try:
    df = pd.read_csv(input_csv, encoding='utf-8')
except UnicodeDecodeError:
    print(f"[!] UTF-8 failed, trying latin1 encoding for '{input_csv}'")
    df = pd.read_csv(input_csv, encoding='latin1')

# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]

# Replace non-breaking spaces (0xA0) with normal spaces in string/object columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.replace('\xa0', ' ', regex=False)

# Optional: remove any invisible/control characters from strings
import re
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].apply(lambda x: re.sub(r'[\x00-\x1F]+', '', str(x)))

# Save fixed CSV
df.to_csv(output_csv, index=False)
print(f"Fixed CSV saved as '{output_csv}'")