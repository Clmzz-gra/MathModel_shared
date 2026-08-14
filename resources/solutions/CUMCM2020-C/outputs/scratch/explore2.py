import pandas as pd, os, sys
base = r'e:\MathModel\problems\CUMCM2020\C'
sys.stdout.reconfigure(encoding='utf-8')

f1 = os.path.join(base, '附件1：123家有信贷记录企业的相关数据.xlsx')
xls = pd.ExcelFile(f1)
print("附件1 sheets:", xls.sheet_names, flush=True)
for s in xls.sheet_names:
    df = pd.read_excel(xls, s)
    print(f"\n[{s}] shape={df.shape}", flush=True)
    print(f"cols: {list(df.columns)}", flush=True)
    print(df.head(2).to_string(max_colwidth=20), flush=True)
