import re
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from table_spliter import split_table  # 假设这是前面实现的表格拆分器

class MarkdownParser:
    def __init__(self, chunk_size=1000, overlap_size=100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
        self.table_pattern = re.compile(r'<html>.*?<body>.*?<table>.*?</table>.*?</body>.*?</html>', re.DOTALL | re.IGNORECASE)

    def parse_markdown_table(self, table_html):
        """解析并拆分HTML表格，返回（子表列表，超大单元格列表）"""
        try:
            # 使用之前实现的表格拆分器
            split_tables, oversized_cells = split_table(table_html, self.chunk_size)
            
            # 转换拆分后的表格结构为字符串
            table_strings = []
            for table in split_tables:
                if table:
                    # 转换为可读性更好的字符串格式
                    header = table[0] if table else []
                    rows = table[1:] if len(table) > 1 else []
                    table_str = "表格内容:\n"
                    table_str += "| " + " | ".join(header) + " |\n"
                    table_str += "| " + " | ".join(["---"]*len(header)) + " |\n"
                    for r in rows:
                        table_str += "| " + " | ".join(r) + " |\n"
                    table_strings.append(table_str.strip())
            
            # 处理超大单元格
            oversized_texts = []
            for cell in oversized_cells:
                parts = cell.split("-", 1)
                if len(parts) == 2:
                    oversized_texts.append(f"超大单元格内容 ({parts[0]}): {parts[1]}")
                else:
                    oversized_texts.append(f"超大单元格内容: {cell}")
            
            return table_strings, oversized_texts
        except Exception as e:
            print(f"表格解析失败: {str(e)}")
            return [], []

    def parse_markdown_to_documents(self, content):
        print("开始解析 Markdown 文档...")
        sections = content.split('\n')
        paragraphs = []
        current_chunk = ""

        for section in sections:
            table_match = self.table_pattern.search(section)
            if table_match:
                table_html = table_match.group(0)
                print("发现表格，进行拆分处理...")
                
                # 解析并拆分表格
                table_strings, oversized_texts = self.parse_markdown_table(table_html)
                
                # 处理当前积累的文本
                if current_chunk:
                    paragraphs.append(current_chunk)
                    current_chunk = ""
                
                # 添加拆分后的表格内容
                paragraphs.extend(table_strings)
                
                # 添加超大单元格到正文
                paragraphs.extend(oversized_texts)

            elif self.heading_pattern.match(section):
                # 标题处理保持原有逻辑
                if current_chunk:
                    paragraphs.append(current_chunk)
                    current_chunk = ""
                current_chunk += section.strip() + "\n"

            else:
                # 普通文本处理
                if section.strip():
                    current_chunk += section.strip() + "\n"

        # 处理最后剩余的文本
        if current_chunk:
            paragraphs.append(current_chunk)

        print("开始分块处理...")
        documents = []
        for paragraph in paragraphs:
            # 合并处理表格和普通文本的分块逻辑
            while len(paragraph) > self.chunk_size:
                # 优先按段落分割
                split_pos = paragraph.rfind('\n', 0, self.chunk_size)
                if split_pos == -1:
                    split_pos = self.chunk_size
                
                chunk = paragraph[:split_pos].strip()
                if chunk:
                    documents.append(Document(page_content=chunk))
                paragraph = paragraph[split_pos:].lstrip()
            
            if paragraph.strip():
                documents.append(Document(page_content=paragraph.strip()))

        print(f"Markdown 解析完成，共生成 {len(documents)} 个文档分块")
        return documents