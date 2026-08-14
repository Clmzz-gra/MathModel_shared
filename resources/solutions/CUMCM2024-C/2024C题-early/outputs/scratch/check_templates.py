import pandas as pd
BASE = r"e:\MathModel\problems\CUMCM2024\CUMCM2024Problems\C题\附件3"
for f in ["result1_1.xlsx", "result1_2.xlsx", "result2.xlsx"]:
    p = f"{BASE}\\{f}"
    df = pd.read_excel(p, header=None)
    print(f"=== {f} ({df.shape}) ===")
    for i in range(min(6, len(df))):
        vs = [str(v)[:40] if pd.notna(v) else "" for v in df.iloc[i].values]
        print(f"  row{i}: {' | '.join(vs)}")
    print()
