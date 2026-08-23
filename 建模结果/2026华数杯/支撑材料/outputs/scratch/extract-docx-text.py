"""
目的：
    提取桌面 docx（队友修改的问题一/问题二文本）为纯文本，供审查

原理：
    docx 是 zip 包，正文在 word/document.xml；用正则切分段落并抽取 <w:t> 文本节点。

输入数据：
    - C:/Users/Lenovo/Desktop/问题一，问题二.docx (原始)

输出：
    逐段打印正文文本（stdout）

对应论文章节：
    问题一/问题二（队友手改稿审查）
"""

import re
import zipfile
from pathlib import Path

p = Path(r"C:\Users\Lenovo\Desktop\问题一，问题二.docx")
with zipfile.ZipFile(p) as z:
    xml = z.read("word/document.xml").decode("utf-8")

for para in re.split(r"</w:p>", xml):
    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para)
    line = "".join(texts).strip()
    if line:
        print(line)
