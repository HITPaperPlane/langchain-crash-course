import os
import re
import csv
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# 导入BeautifulSoup库用于解析HTML
from bs4 import BeautifulSoup

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag_gemini") # 请根据你的实际情况修改数据库路径
# persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag") # 如果要用之前的数据库，取消注释这行，并注释上一行

# 定义嵌入模型
embeddings = OllamaEmbeddings(model="qwen2.5")

# 检查数据库是否已经存在
if os.path.exists(persistent_directory):
    print("数据库已经存在，加载现有数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("数据库不存在，正在创建新的数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 创建检索器来查询向量存储
retriever = db.as_retriever(
    search_type="mmr",  # 使用MMR策略进行检索
    search_kwargs={"k": 3, "lambda_mult": 0.8}  # lambda_mult 用来平衡相关性与多样性
)
# 创建ChatOpenAI模型
llm = ChatOpenAI(model="gpt-4o")

# 上下文化问题提示
contextualize_q_system_prompt = (
    "给定一个聊天历史和最新的用户问题，"
    "该问题可能引用了聊天历史中的内容，"
    "重新表述问题使其成为一个独立的、可以理解的问题。"
    "如果不需要修改问题，则原样返回，不要回答问题。"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 创建一个历史感知型检索器 (这里移除了压缩检索器，直接使用原始检索器)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt #  使用原始 retriever，移除 compression_retriever
)

# 问答提示
qa_system_prompt = (
    "你是一个问答助手。利用以下检索到的上下文来回答问题，"
    "如果你不知道答案，就说不知道。请保持答案简洁，最多三句话。"
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
)

# 创建问答链
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 创建检索链，结合历史感知型检索器和问答链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def parse_markdown_table(table_html, chunk_size):
    """
    解析HTML表格并尝试结构化表示，如果超过chunk_size则返回None。
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

    if len(table_string) <= chunk_size:
        return table_list, table_string  # 返回列表形式和字符串形式
    else:
        return None, None # 表格过大，返回None


def decompose_markdown_table(table_html):
    """
    分解Markdown表格为行标题、列标题和单元格信息，并包含行列标题信息在单元格中。
    改进：显式处理行标题，并将行列标题信息包含在单元格数据中。
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return {}

    column_headers = [th.text.strip() for th in table.find_all('th')]
    row_headers = []
    table_data = []

    body = table.find('tbody') if table.find('tbody') else table # 应对没有tbody的情况
    for row_index, tr in enumerate(body.find_all('tr')):
        cells = tr.find_all('td')
        th_cells = tr.find_all('th') # 查找行标题 th 元素

        if th_cells:
            row_header_text = th_cells[0].text.strip() # 假设第一个 th 是行标题
            row_headers.append(row_header_text)
        else:
            row_headers.append(f"行 {row_index+1}") # 如果没有 th，则使用默认行标题，例如 "行 1", "行 2" ...

        if not cells and not column_headers: # 处理只有表头没有数据行的情况
            continue

        row_data = {}
        for col_index, td in enumerate(cells):
            cell_content = td.text.strip()
            col_header = column_headers[col_index] if column_headers and col_index < len(column_headers) else f"列 {col_index+1}"
            row_header = row_headers[-1] if row_headers else f"行 {row_index+1}" # 获取当前行标题

            # 单元格信息包含行列标题
            cell_info = {
                "cell_content": cell_content,
                "column_header": col_header,
                "row_header": row_header,
            }
            row_data[col_header] = cell_info # 以列标题作为键，存储单元格的详细信息

        table_data.append(row_data)

    return {
        "column_headers": column_headers,
        "row_headers": row_headers,
        "table_data": table_data
    }



# 解析Markdown文件并返回Document对象的列表
def parse_markdown_to_documents(content, chunk_size=1000, overlap_size=100):
    print("开始解析 Markdown 文档...") # 添加日志
    heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
    table_pattern = re.compile(r'<html><body><table>.*?</table></body></html>', re.DOTALL) # 匹配<table>标签
    documents = []
    current_titles = []  # 当前层级的标题列表
    sections = content.split('\n')
    chunk = ""
    paragraphs = []

    for section in sections:
        is_table_section = False # 标记当前section是否为表格

        table_match = table_pattern.search(section)
        if table_match:
            table_html = table_match.group(0)
            is_table_section = True
            print("尝试整体解析表格...") # 添加日志
            # 尝试将整个表格作为一个块
            table_list, table_string = parse_markdown_table(table_html, chunk_size)
            if table_list:
                if chunk: # 先处理之前的chunk
                    paragraphs.append(chunk)
                    chunk = ""
                paragraphs.append(table_string) # 将表格的字符串表示加入段落
            else:
                print("整体解析表格失败，尝试分解表格...") # 添加日志
                # 如果表格过大，则分解表格
                decomposed_table = decompose_markdown_table(table_html)
                if decomposed_table:
                    if chunk: # 先处理之前的chunk
                        paragraphs.append(chunk)
                        chunk = ""
                    # 将分解的表格信息加入段落 (这里将表格信息转换为更易于理解和检索的文本描述)
                    paragraphs.append(f"表格列标题: {decomposed_table['column_headers']}")
                    if decomposed_table['row_headers']:
                        paragraphs.append(f"表格行标题: {decomposed_table['row_headers']}") # 添加行标题
                    for row_data in decomposed_table['table_data']:
                        # 针对每个单元格，包含更丰富的行列标题信息
                        row_text_parts = []
                        for col_header, cell_info in row_data.items():
                            row_text_parts.append(f"| {cell_info['column_header']}: {cell_info['cell_content']} (行: {cell_info['row_header']}) ")
                        paragraphs.append("表格行数据: " + "".join(row_text_parts)) # 将行数据合并为一个段落

            continue # 表格 section 已经处理完毕, continue


        heading_match = heading_pattern.match(section)
        if heading_match:
            if chunk:
                paragraphs.append(chunk)
            current_depth = len(heading_match.group(1)) - 1
            page_content = heading_match.group(2).strip()
            current_titles = current_titles[:current_depth]
            current_titles.append(page_content)
            chunk = ""
            print("解析到标题：", current_titles) # 添加日志
        elif not is_table_section: # 非表格内容才累加到 chunk，表格内容已经被处理了
            if section.strip():
                chunk += section.strip() + "\n"


    if chunk: # 处理最后一个chunk
        paragraphs.append(chunk)

    print("解析出的段落 (chunking 前):", paragraphs) # 添加日志

    documents = []
    for i in range(len(paragraphs)):
        content = paragraphs[i]
        print("处理段落:", content) # 添加日志
        while len(content) > chunk_size:
            doc_chunk = content[:chunk_size]
            content = content[chunk_size - overlap_size:]
            full_content = ' > '.join(current_titles) + "\n" + doc_chunk.strip()
            documents.append(Document(page_content=full_content, metadata={"context_title": ' > '.join(current_titles)}))

        if content.strip():
            full_content = ' > '.join(current_titles) + "\n" + content.strip()
            documents.append(Document(page_content=full_content, metadata={"context_title": ' > '.join(current_titles)}))

    print("Markdown 文档解析完成，共生成 {} 个文档分块".format(len(documents))) # 添加日志
    return documents



# 合并标题和对应的内容 (此函数目前未使用，可以根据需要调整)
def merge_title_content(data):
    merged_data = []
    current_title = None
    current_content = []

    for document in data:
        metadata = document.metadata
        category_depth = metadata.get('category_depth', None)
        context_title = metadata.get('context_title', None)
        page_content = document.page_content

        if category_depth == 0:
            if current_title is not None:
                merged_content = "\n".join(current_content)
                merged_data.append({
                    'title': current_title,
                    'content': merged_content
                })
                current_content = []
            current_title = f"{context_title} {page_content}"

        elif category_depth is not None:
            current_content.append(f"{context_title} {page_content}")
        else:
            current_content.append(page_content)

    if current_title is not None:
        merged_content = "\n".join(current_content)
        merged_data.append({
            'title': current_title,
            'content': merged_content
        })

    return merged_data


# 获取指定文件夹下所有.md文件的内容并将chunks存入数据库
# 解析并存储Markdown文件，若数据库已存在则不重新创建
def parse_and_store_markdown_files(folder_path, chunk_size=1000, overlap_size=100):
    print("开始解析和存储 Markdown 文件...") # 添加日志
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    all_documents = []

    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        print("读取 Markdown 文件:", file_name) # 添加日志
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = parse_markdown_to_documents(content, chunk_size, overlap_size)
            print("文件 {} 解析完成，生成 {} 个文档分块，准备存储到数据库...".format(file_name, len(documents))) # 添加日志

            # 将内容分块并存储到数据库
            for doc in documents:
                chunk = doc.page_content
                embedding = embeddings.embed_query(chunk)
                db.add_texts([chunk], embeddings=[embedding], metadatas=[doc.metadata])
            print("成功存储 {} 个文档分块到数据库".format(len(documents))) # 添加日志

        all_documents.extend(documents)

    db.persist()
    print("所有 Markdown 文件解析和存储完成，数据库持久化完成。") # 添加日志
    return all_documents


# 打印数据库中的所有文本分块
def print_database_chunks():
    print("\n--- 当前数据库中的分块 ---")
    all_docs = db.get()
    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    csv_file = "database_chunks_gemini.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "文本内容", "元数据"])

        for idx, (doc, metadata) in enumerate(zip(docs, metadatas), 1):
            print(f"分块 {idx}: {doc}")
            print(f"元数据{idx}: {metadata}")
            writer.writerow([idx, doc, str(metadata)])

    print(f"\n分块数据已保存到 {csv_file}")



# 模拟持续对话
def continual_chat():
    print("开始与AI聊天！输入'exit'结束对话。")
    chat_history = []
    while True:
        query = input("你: ")
        if query.lower() == "exit":
            break
        print("用户提问：{}".format(query)) # 添加日志
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        context = result['context']
        print("检索到的上下文 (invoke 结果): {}".format(context)) # 添加日志

        print(f"\nAI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


# 主函数
if __name__ == "__main__":
    # 解析并存储Markdown文件，若数据库已存在则不重新创建
    markdown_folder = "Try-Markdown-RAG/markdown"
    documents = parse_and_store_markdown_files(markdown_folder, chunk_size=1000, overlap_size=100) # 减小chunk_size以便测试表格拆分
    merged_data = merge_title_content(documents) # 此函数目前没有使用

    # 输出当前数据库中的文本分块
    print_database_chunks()

    # 启动对话功能
    continual_chat()