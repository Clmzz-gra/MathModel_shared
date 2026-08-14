import pandas as pd
import os

base = r'e:\MathModel\problems\CUMCM2020\C'

# 附件1
print("=== 附件1 ===")
f1 = os.path.join(base, '附件1：123家有信贷记录企业的相关数据.xlsx')
xls1 = pd.ExcelFile(f1)
for s in xls1.sheet_names:
    df = pd.read_excel(xls1, s)
    print(f"Sheet [{s}]: shape={df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Head:\n{df.head(3).to_string()}\n")

# 附件2
print("=== 附件2 ===")
f2 = os.path.join(base, '附件2：302家无信贷记录企业的相关数据.xlsx')
xls2 = pd.ExcelFile(f2)
for s in xls2.sheet_names:
    df = pd.read_excel(xls2, s)
    print(f"Sheet [{s}]: shape={df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Head:\n{df.head(3).to_string()}\n")

# 附件3
print("=== 附件3 ===")
f3 = os.path.join(base, '附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
df3 = pd.read_excel(f3)
print(f"shape={df3.shape}")
print(f"Columns: {list(df3.columns)}")
print(f"Head:\n{df3.head(10).to_string()}")
