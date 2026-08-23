"""
目的：
    提取《问题一.docx》全文（段落+表格），用于与 chapter4-sub1.tex 对比合并

原理：
    python-docx 遍历 document.paragraphs 与 tables，输出为纯文本（保留段落顺序）

输入数据：
    - 问题一.docx (原始, 位于 C:/Users/Lenovo/Downloads/)

输出：
    控制台打印文本

对应论文章节：
    §四 问题一
"""
import sys
import docx
from docx import Document

path = r"C:\Users\Lenovo\Downloads\问题一.docx"
doc = Document(path)

# 遍历文档主体元素（段落与表格按顺序）
from docx.oxml.ns import qn

body = doc.element.body
for child in body.iterchildren():
    if child.tag == qn('w:p'):
        p = docx.text.paragraph.Paragraph(child, doc)
        t = p.text.strip()
        if t:
            print(t)
    elif child.tag == qn('w:tbl'):
        tbl = docx.table.Table(child, doc)
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells]
            print(" | ".join(cells))
