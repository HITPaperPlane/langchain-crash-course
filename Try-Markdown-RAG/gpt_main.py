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
from bs4 import BeautifulSoup

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag_gpt")

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

# 创建一个历史感知型检索器
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
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

# 创建检索链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def generate_natural_language_description(prompt, llm, input_text):
    """
    使用LLM生成自然语言描述。
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=input_text)
    ]
    # prompt_template = ChatPromptTemplate.from_messages(messages)
    response = llm.invoke(messages)
    return response.content


def table_to_natural_language(table_html, chunk_size, llm):
    """
    将HTML表格转换为自然语言描述，并进行分块处理。利用LLM来生成描述。
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    headers = [th.text.strip() for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr'):
        row_data = [td.text.strip() for td in tr.find_all('td')]
        if row_data:
            rows.append(row_data)

    documents = []

    # 1. 描述行标题 (使用LLM)
    #  更健壮的检查： 确保有行数据 *并且* 第一行有数据
    if rows and rows[0]:
        print("正在描述行标题")
        row_headers = [row[0] for row in rows]
        prompt = "这是一个表格的行标题，请用自然语言描述一下，要求简洁明了："
        row_header_desc = generate_natural_language_description(prompt, llm, "，".join(row_headers))
        documents.append(row_header_desc)

    # 2. 描述列标题 (使用LLM)
    # 更健壮的检查：确保headers列表非空 *并且* 长度大于1
    if headers and len(headers) > 1:
        print("正在描述列标题")
        prompt = "这是一个表格的列标题，请用自然语言描述一下，要求简洁明了："
        col_header_desc = generate_natural_language_description(prompt, llm, "，".join(headers[1:])) #跳过第一个，那是行标题
        documents.append(col_header_desc)


    # 3. 描述每个单元格 (使用LLM)
    # 更健壮的检查： 确保headers和rows列表都非空
    if headers and rows:
        for i, row in enumerate(rows):
            for j, cell in enumerate(row[1:], 1):  # 从第二个单元格开始
                print("正在描述单元格")
                prompt = (
                    f"这是一个表格中的一个单元格，其行标题是'{row[0]}', 列标题是'{headers[j]}',"
                    f"请用自然语言描述这个单元格的内容：'{cell}'"
                )
                cell_desc = generate_natural_language_description(prompt, llm, "") #prompt 已经包含了信息
                documents.append(cell_desc)

    # 分块
    chunked_docs = []
    current_chunk = ""
    for doc in documents:
        if len(current_chunk) + len(doc) + 1 <= chunk_size:
            current_chunk += doc + "\n"
        else:
            if current_chunk:
                chunked_docs.append(current_chunk.strip())
            current_chunk = doc + "\n"
    if current_chunk:
        chunked_docs.append(current_chunk.strip())

    return chunked_docs



# 解析Markdown文件并返回Document对象的列表
def parse_markdown_to_documents(content, chunk_size=1000, overlap_size=100):
    print("开始解析 Markdown 文档...")
    heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
    table_pattern = re.compile(r'<html><body><table>.*?</table></body></html>', re.DOTALL)  # 匹配<table>标签

    documents = []
    current_titles = []  # 当前层级的标题列表
    sections = content.split('\n')
    chunk = ""


    for section in sections:

        table_match = table_pattern.search(section)
        if table_match:
            # 处理表格
            if chunk: # 结算非表格内容
                documents.append(Document(page_content=' > '.join(current_titles) + "\n" + chunk.strip(), metadata={"context_title": ' > '.join(current_titles)}))
                chunk = ""

            table_html = table_match.group(0)
            print("解析到表格,转换为自然语言...")
            table_docs = table_to_natural_language(table_html, chunk_size, llm)

            for table_doc_str in table_docs:
                # 表格的每个自然语言描述作为一个单独的document
                documents.append(Document(page_content=' > '.join(current_titles) + "\n" + table_doc_str, metadata={"context_title": ' > '.join(current_titles)}))
            continue

        heading_match = heading_pattern.match(section)
        if heading_match:
            # 结算之前的chunk
            if chunk:
                documents.append(Document(page_content=' > '.join(current_titles) + "\n" + chunk.strip(), metadata={"context_title": ' > '.join(current_titles)}))


            current_depth = len(heading_match.group(1)) - 1
            page_content = heading_match.group(2).strip()
            current_titles = current_titles[:current_depth]
            current_titles.append(page_content)
            chunk = ""  # Reset chunk
            print("解析到标题：", current_titles)

        else: # 非表格，非标题
            if section.strip():
                chunk += section.strip() + "\n"

    # 最后一个chunk
    if chunk:
        documents.append(Document(page_content=' > '.join(current_titles) + "\n" + chunk.strip(), metadata={"context_title": ' > '.join(current_titles)}))

    print("Markdown 文档解析完成，共生成 {} 个文档分块".format(len(documents)))
    return documents



# 获取指定文件夹下所有.md文件的内容并将chunks存入数据库
def parse_and_store_markdown_files(folder_path, chunk_size=1000, overlap_size=100):
    print("开始解析和存储 Markdown 文件...")
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    all_documents = []

    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        print("读取 Markdown 文件:", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = parse_markdown_to_documents(content, chunk_size, overlap_size)
            print("文件 {} 解析完成，生成 {} 个文档分块，准备存储到数据库...".format(file_name, len(documents)))

            # 将内容分块并存储到数据库
            for doc in documents:
                chunk = doc.page_content
                embedding = embeddings.embed_query(chunk)
                db.add_texts([chunk], embeddings=[embedding], metadatas=[doc.metadata])
            print("成功存储 {} 个文档分块到数据库".format(len(documents)))

        all_documents.extend(documents)

    db.persist()
    print("所有 Markdown 文件解析和存储完成，数据库持久化完成。")
    return all_documents


# 打印数据库中的所有文本分块
def print_database_chunks():
    print("\n--- 当前数据库中的分块 ---")
    all_docs = db.get()
    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    csv_file = "database_chunks_gpt.csv"
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
        print("用户提问：{}".format(query))
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        context = result['context']
        print("检索到的上下文 (invoke 结果): {}".format(context))

        print(f"\nAI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


# 主函数
if __name__ == "__main__":
    # 解析并存储Markdown文件，若数据库已存在则不重新创建
    markdown_folder = "Try-Markdown-RAG/markdown"
    documents = parse_and_store_markdown_files(markdown_folder, chunk_size=1000, overlap_size=100)
    
    # 输出当前数据库中的文本分块
    print_database_chunks()

    # 启动对话功能
    continual_chat()