# test_preprocessing.py
from app.services.ml.preprocessing import run_preprocessing

with open("sample_data.csv", "rb") as f:
    file_bytes = f.read()

result = run_preprocessing(file_bytes, "text/csv")

print("=== Summary ===")
for key, val in result.summary.items():
    print(f"  {key}: {val}")

print(f"\n=== RFM Table (first 5 rows) ===")
print(result.df_rfm.head())

print(f"\n=== Basket Table (first 3 rows) ===")
print(result.df_basket.head(3))

print(f"\n=== Column mapping ===")
print(result.column_map)