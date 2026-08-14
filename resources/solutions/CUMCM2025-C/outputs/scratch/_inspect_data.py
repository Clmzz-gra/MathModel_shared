# Quick data inspection
import pandas as pd

for name, path in [
    ("Male", "2025C-male-clean.pkl"),
    ("Female", "2025C-female-clean.pkl"),
]:
    base = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/data"
    df = pd.read_pickle(f"{base}/{path}")
    print(f"\n{'='*60}")
    print(f"{name}: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if "bmi_group" in df.columns:
        print(f"bmi_group: {df.bmi_group.value_counts().to_dict()}")
    if "AB_异常" in df.columns:
        print(f"AB_异常: {df.AB_异常.value_counts().to_dict()}")
    if "孕妇ID" in df.columns or "ID" in df.columns:
        col = "孕妇ID" if "孕妇ID" in df.columns else "ID"
        print(f"Unique {col}: {df[col].nunique()}")
