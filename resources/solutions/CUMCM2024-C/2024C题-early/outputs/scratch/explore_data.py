"""阶段 0.2 数据盘点：探索附件 Excel 结构"""
import pandas as pd
from pathlib import Path

BASE = Path(r"e:\MathModel\problems\CUMCM2024\CUMCM2024Problems\C题")
OUT = Path(r"e:\MathModel\problems\CUMCM2024\C题\outputs\data")
OUT.mkdir(parents=True, exist_ok=True)

attachments = {
    "附件1": BASE / "附件1.xlsx",
    "附件2": BASE / "附件2.xlsx",
}

for name, path in attachments.items():
    print(f"\n{'='*60}")
    print(f"  {name}: {path.name}")
    print(f"{'='*60}")
    xl = pd.ExcelFile(path)
    print(f"  Sheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        print(f"\n  --- Sheet: {sheet} ---")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
        # 打印前5行看结构
        print(f"  First 5 rows:")
        for i in range(min(8, len(df))):
            vals = [str(v)[:50] if pd.notna(v) else "NaN" for v in df.iloc[i].values]
            print(f"    row{i}: {' | '.join(vals)}")

# 检查附件3
att3_dir = BASE.parent  # CUMCM2024Problems
print(f"\n{'='*60}")
print(f"  搜索附件3...")
print(f"{'='*60}")
for f in sorted(BASE.parent.rglob("*.xlsx")):
    if "result" in f.name.lower() or "附件3" in str(f):
        print(f"  Found: {f}")
