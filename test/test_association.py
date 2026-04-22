from app.services.ml.preprocessing import run_preprocessing
from app.services.ml.association_rules import run_association_rules

with open("sample_data.csv", "rb") as f:
    file_bytes = f.read()

prep   = run_preprocessing(file_bytes, "text/csv")
result = run_association_rules(prep.df_basket)

print(f"Total rules found: {result.total_found}")
print(f"Showing top {len(result.rules)} rules\n")
print("=== Top Association Rules ===\n")

for rule in result.rules:
    ant  = " + ".join(rule["antecedents"])
    con  = " + ".join(rule["consequents"])
    conf = rule["confidence"] * 100
    lift = rule["lift"]
    print(f"  {ant}  →  {con}")
    print(f"    confidence: {conf:.0f}%   lift: {lift:.2f}x")
    print()