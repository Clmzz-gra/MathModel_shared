import pandas as pd
df = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-male-clean.pkl")
print("All columns:")
for i, c in enumerate(df.columns):
    print(f"  [{i}] '{c}'")
print(f"\nContains 'ID': {[c for c in df.columns if 'ID' in c]}")
print(f"Contains 'bmi': {[c for c in df.columns if 'bmi' in c.lower()]}")
print(f"Contains '13': {[c for c in df.columns if '13' in c]}")
print(f"Contains 'IVF': {[c for c in df.columns if 'IVF' in c]}")
print(f"\nUnique IDs: {df[[c for c in df.columns if 'ID' in c][0]].nunique() if [c for c in df.columns if 'ID' in c] else 'N/A'}")

print("\n\n=== Female ===")
df2 = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-female-clean.pkl")
print(f"Columns: {df2.columns.tolist()}")
print(f"Contains 'AB': {[c for c in df2.columns if 'AB' in c]}")
print(f"Contains 'ID': {[c for c in df2.columns if 'ID' in c]}")
print(f"Contains '13': {[c for c in df2.columns if '13' in c]}")
