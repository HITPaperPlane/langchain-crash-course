import re
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from table_spliter import split_table

class MarkdownParser:
    def __init__(self, chunk_size=1000, overlap_size=100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
        self.table_pattern = re.compile(r'<html>.*?<body>.*?<table>.*?</table>.*?</body>.*?</html>', re.DOTALL | re.IGNORECASE)
        self.current_headings = []  # 标题层级栈
        self.active_headings = []   # 有效标题缓存

    def _update_headings(self, level: str, title: str):
        """智能更新标题层级"""
        # 清理无效字符
        title = title.strip().replace('\n', ' ')
        
        # 计算当前标题深度
        current_depth = len(level) - 1
        
        # 修正层级异常
        if current_depth > len(self.current_headings):
            # 补齐缺失的中间层级
            for _ in range(current_depth - len(self.current_headings)):
                self.current_headings.append("未命名标题")
        
        # 截断到当前深度
        self.current_headings = self.current_headings[:current_depth]
        
        # 添加/更新当前层级标题
        if len(self.current_headings) == current_depth:
            self.current_headings.append(title)
        else:
            self.current_headings[current_depth] = title
        
        # 缓存有效标题（长度校验）
        self.active_headings = []
        current_prefix = ""
        for i, h in enumerate(self.current_headings):
            new_prefix = f"{'#'*(i+1)} {h}"
            if len(current_prefix) + len(new_prefix) + 2 <= self.chunk_size:
                current_prefix += f" {new_prefix}" if current_prefix else new_prefix
                self.active_headings.append(h)
            else:
                break

    def _get_heading_prefix(self):
        """生成安全的标题前缀"""
        if not self.active_headings:
            return ""
        return '  '.join([f"{'#'*(i+1)} {h}" for i, h in enumerate(self.active_headings)])

    def parse_markdown_table(self, table_html):
        """解析表格并保持当前标题上下文"""
        try:
            # 保存当前标题状态
            original_headings = self.current_headings.copy()
            original_active = self.active_headings.copy()
            
            split_tables, oversized_cells = split_table(table_html, self.chunk_size)
            
            # 恢复标题状态
            self.current_headings = original_headings
            self.active_headings = original_active
            
            return (
                [str(t) for t in split_tables if t],
                [f"{h[0]}-{h[1]}" if isinstance(h, tuple) else h for h in oversized_cells]
            )
        except Exception as e:
            print(f"表格解析失败: {str(e)}")
            return [], []

    def parse_markdown_to_documents(self, content):
        print("开始解析 Markdown 文档...")
        sections = content.split('\n')
        documents = []
        current_chunk = ""
        current_prefix = ""

        for section in sections:
            # 处理表格
            if (table_match := self.table_pattern.search(section)):
                table_html = table_match.group(0)
                print("发现表格，进行拆分处理...")
                
                # 处理表格前提交当前块
                if current_chunk:
                    documents.append(current_prefix + " " + current_chunk.strip())
                    current_chunk = ""
                
                tables, cells = self.parse_markdown_table(table_html)
                
                # 添加带标题的表格内容
                for table in tables:
                    full_content = f"{current_prefix} {table}".strip()
                    if len(full_content) <= self.chunk_size:
                        documents.append(full_content)
                    else:
                        print(f"表格内容过长（{len(full_content)}字符），将进行分块处理")
                
                # 处理超大单元格
                for cell in cells:
                    full_cell = f"{current_prefix} {cell}".strip()
                    if len(full_cell) <= self.chunk_size:
                        documents.append(full_cell)
                    else:
                        print(f"超大单元格内容过长（{len(full_cell)}字符），已丢弃")

            # 处理标题
            elif (heading_match := self.heading_pattern.match(section)):
                # 提交当前块
                if current_chunk:
                    documents.append(current_prefix + " " + current_chunk.strip())
                    current_chunk = ""
                
                level, title = heading_match.groups()
                self._update_headings(level, title)
                current_prefix = self._get_heading_prefix()
                
                # 标题自身作为独立块处理
                if current_prefix and len(current_prefix) <= self.chunk_size:
                    documents.append(current_prefix)
                elif current_prefix:
                    print(f"标题过长被截断（{len(current_prefix)}字符）")

            # 处理普通内容
            elif section.strip():
                # 智能拼接内容
                new_content = section.strip()
                chunk_candidate = f"{current_chunk} {new_content}".strip() if current_chunk else new_content
                
                if len(chunk_candidate) <= self.chunk_size - len(current_prefix) - 1:
                    current_chunk = chunk_candidate
                else:
                    if current_chunk:
                        documents.append(current_prefix + " " + current_chunk)
                        current_chunk = new_content
                    else:
                        # 处理超长单行内容
                        while len(new_content) > 0:
                            take = self.chunk_size - len(current_prefix) - 1
                            documents.append(current_prefix + " " + new_content[:take])
                            new_content = new_content[take:]

        # 处理最后的内容
        if current_chunk:
            documents.append(current_prefix + " " + current_chunk.strip())

        print("开始最终分块处理...")
        final_chunks = []
        for doc in documents:
            # 最终长度校验
            while len(doc) > self.chunk_size:
                split_pos = doc.rfind(' ', 0, self.chunk_size)
                if split_pos == -1:
                    split_pos = self.chunk_size
                final_chunks.append(doc[:split_pos])
                doc = doc[split_pos:].lstrip()
            if doc:
                final_chunks.append(doc)

        print(f"生成 {len(final_chunks)} 个有效分块")
        return [Document(page_content=c) for c in final_chunks if c.strip()]