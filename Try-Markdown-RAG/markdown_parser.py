import re
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

class MarkdownParser:
    def __init__(self, chunk_size=1000, overlap_size=100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
        # 修改表格匹配模式，使其更宽松
        self.table_pattern = re.compile(r'<html>.*?<body>.*?<table>.*?</table>.*?</body>.*?</html>', re.DOTALL | re.IGNORECASE)
        # self.table_pattern = re.compile(r'<html><body><table>.*?</table></body></html>', re.DOTALL)

    def parse_markdown_table(self, table_html):
        """
        解析HTML表格并尝试结构化表示，如果超过chunk_size则报错.
        """
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        if not table:
            return None

        headers = [th.text.strip() for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr'):
            row_data = [td.text.strip() for td in tr.find_all('td')]
            if row_data:  # 确保行不为空
                rows.append(row_data)

        # 构建表格的列表形式
        table_list = [headers] + rows if headers else rows
        table_string = str(table_list)

        if len(table_string) <= self.chunk_size:
            return table_list, table_string  # 返回列表形式和字符串形式
        else:
            raise ValueError("表格过大，无法装入一个chunk。")


    def parse_markdown_to_documents(self, content):

        print("开始解析 Markdown 文档...")  # 添加日志
        sections = content.split('\n')
        paragraphs = []
        current_chunk = ""

        for section in sections:
            table_match = self.table_pattern.search(section)
            if table_match:
                table_html = table_match.group(0)
                print("尝试解析表格...")  # 添加日志
                # 尝试将整个表格作为一个块
                try:
                    table_list, table_string = self.parse_markdown_table(table_html)
                    if current_chunk:
                        paragraphs.append(current_chunk)
                        current_chunk = ""
                    paragraphs.append(table_string)  # 直接添加表格的字符串表示
                except ValueError as e:
                    print(str(e))  # 打印错误信息
                    raise  # 重新抛出异常，停止程序


            elif self.heading_pattern.match(section):
                if current_chunk:
                    paragraphs.append(current_chunk)
                    current_chunk = ""
                # 不再处理标题，直接添加到chunk
                current_chunk += section.strip() + "\n"

            else:
                if section.strip():
                    current_chunk += section.strip() + "\n"


        if current_chunk:  # 处理最后一个chunk
            paragraphs.append(current_chunk)


        print("解析出的段落 (chunking 前):", paragraphs)  # 添加日志

        documents = []
        for paragraph in paragraphs:
            while len(paragraph) > self.chunk_size:
                doc_chunk = paragraph[:self.chunk_size]
                paragraph = paragraph[self.chunk_size - self.overlap_size:]
                documents.append(Document(page_content=doc_chunk.strip()))
            if paragraph.strip():
                documents.append(Document(page_content=paragraph.strip()))

        print("Markdown 文档解析完成，共生成 {} 个文档分块".format(len(documents)))  # 添加日志
        return documents