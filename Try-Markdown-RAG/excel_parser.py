import os
import pandas as pd
from langchain.docstore.document import Document

class ExcelParser:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def parse_excel_to_documents(self, file_path):
        """解析 Excel 文件并生成文档分块"""
        print(f"开始解析 Excel 文件: {file_path}")
        
        # 读取 Excel 文件
        df = pd.read_excel(file_path, sheet_name=None)  # 读取所有表单
        
        documents = []
        
        # 遍历每个 sheet
        for sheet_name, sheet_data in df.items():
            print(f"解析工作表: {sheet_name}")
            
            # 将每一行转换为块，每个块保留行和列标题信息
            for index, row in sheet_data.iterrows():
                chunk_content = self._generate_chunk_from_row(sheet_data.columns, row)
                if len(chunk_content) <= self.chunk_size:
                    documents.append(Document(page_content=chunk_content))
                else:
                    print(f"行内容过长（{len(chunk_content)}字符），将进行分块处理")
                    # 如果内容过长，则根据 chunk_size 分割
                    documents.extend(self._split_long_content(chunk_content))
                    
        print(f"共生成 {len(documents)} 个文档分块")
        return documents

    def _generate_chunk_from_row(self, columns, row):
        """生成每一行的内容块"""
        chunk = ""
        for col, value in row.items():
            chunk += f"{col}: {value}  "
        return chunk.strip()

    def _split_long_content(self, content):
        """将长内容根据 chunk_size 分割成多个块"""
        chunks = []
        while len(content) > self.chunk_size:
            split_pos = content.rfind(' ', 0, self.chunk_size)
            if split_pos == -1:
                split_pos = self.chunk_size
            chunks.append(content[:split_pos])
            content = content[split_pos:].lstrip()
        if content:
            chunks.append(content)
        return chunks
