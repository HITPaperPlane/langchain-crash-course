import os
import re
import csv
import numpy as np
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag_ds")

# 定义嵌入模型
embeddings = OllamaEmbeddings(model="qwen2.5")

# 初始化重排序模型
reranker = CrossEncoder('BAAI/bge-reranker-large')

# 检查数据库是否已经存在
if os.path.exists(persistent_directory):
    print("数据库已经存在，加载现有数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("数据库不存在，正在创建新的数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 创建检索器来查询向量存储（使用较大k值便于后续重排序）
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "lambda_mult": 0.8}  # 初始检索较多结果用于重排序
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
    "如果上下文包含表格数据，请特别注意保持数据的完整性。"
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

def parse_table(table, current_titles):
    """解析HTML表格并生成结构化文档块"""
    documents = []
    
    # 提取表头
    headers = []
    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    
    # 提取数据行
    rows = []
    for row in table.find_all('tr')[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)
    
    # 构建表格文本
    table_text = "|".join(headers) + "\n"
    table_text += "|".join(["---"]*len(headers)) + "\n"
    for row in rows:
        table_text += "|".join(row) + "\n"
    
    # 合并标题
    title_str = ' > '.join(current_titles)
    
    # 分块处理
    chunk_size = 2000
    overlap_size = 400
    current_chunk = ""
    
    # 添加表头到每个分块
    header_content = f"{title_str}\n表格:\n|".join(headers) + "\n" + "|".join(["---"]*len(headers)) + "\n"
    max_content_size = chunk_size - len(header_content)
    
    for row_text in ["|".join(row)+"\n" for row in rows]:
        if len(current_chunk) + len(row_text) > max_content_size:
            # 保存当前块
            full_chunk = f"{title_str}\n表格:\n{header_content}{current_chunk}"
            documents.append(Document(
                page_content=full_chunk.strip(),
                metadata={"context_title": title_str, "type": "table"}
            ))
            current_chunk = row_text
            max_content_size = chunk_size - len(header_content) - overlap_size
        else:
            current_chunk += row_text
    
    # 处理最后一块
    if current_chunk:
        full_chunk = f"{title_str}\n表格:\n{header_content}{current_chunk}"
        documents.append(Document(
            page_content=full_chunk.strip(),
            metadata={"context_title": title_str, "type": "table"}
        ))
    
    return documents

def parse_markdown_to_documents(content, chunk_size=2000, overlap_size=400):
    """改进的Markdown解析函数，包含表格处理"""
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    
    # 处理非表格内容
    non_table_soup = soup.copy()
    for table in tables:
        table.decompose()
    non_table_content = str(non_table_soup)
    
    documents = []
    current_titles = []
    current_chunk = ""
    
    # 解析非表格内容
    lines = non_table_content.split('\n')
    for line in lines:
        heading_match = re.match(r'^(#+)\s*(.*)$', line)
        if heading_match:
            if current_chunk:
                documents.append(Document(
                    page_content=' > '.join(current_titles) + "\n" + current_chunk.strip(),
                    metadata={"context_title": ' > '.join(current_titles)}
                ))
                current_chunk = ""
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_titles = current_titles[:level-1]
            current_titles.append(title)
        else:
            if line.strip():
                current_chunk += line.strip() + "\n"
                if len(current_chunk) > chunk_size:
                    # 分割大块文本
                    while len(current_chunk) > chunk_size:
                        split_index = chunk_size
                        # 查找最近的句末分割点
                        for punct in ['.', '!', '?', '\n\n']:
                            last_punct = current_chunk.rfind(punct, 0, split_index)
                            if last_punct != -1:
                                split_index = last_punct + 1
                                break
                        documents.append(Document(
                            page_content=' > '.join(current_titles) + "\n" + current_chunk[:split_index].strip(),
                            metadata={"context_title": ' > '.join(current_titles)}
                        ))
                        current_chunk = current_chunk[split_index - overlap_size:]
    
    # 处理最后一块非表格内容
    if current_chunk.strip():
        documents.append(Document(
            page_content=' > '.join(current_titles) + "\n" + current_chunk.strip(),
            metadata={"context_title": ' > '.join(current_titles)}
        ))
    
    # 处理表格内容
    for table in tables:
        table_docs = parse_table(table, current_titles)
        documents.extend(table_docs)
    
    return documents

# ... [其他辅助函数保持不变，包括merge_title_content, parse_and_store_markdown_files, print_database_chunks] ...

# 合并标题和对应的内容
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
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    all_documents = []

    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = parse_markdown_to_documents(content, chunk_size, overlap_size)

            # 将内容分块并存储到数据库
            for doc in documents:
                chunk = doc.page_content
                # 为每个块生成嵌入并存储到数据库
                embedding = embeddings.embed_query(chunk)  # 使用 embed_query 替换 embed_text
                db.add_texts([chunk], embeddings=[embedding], metadatas=[doc.metadata])
                
            all_documents.extend(documents)

    db.persist()  # 保存所有变化到数据库
    return all_documents


# 打印数据库中的所有文本分块
def print_database_chunks():
    print("\n--- 当前数据库中的分块 ---")
    # 获取数据库中的所有文本
    all_docs = db.get()
    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    # 创建CSV文件
    csv_file = "database_chunks_ds.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "文本内容", "元数据"])
        
        # 使用 zip 来同时遍历 docs 和 metadatas
        for idx, (doc, metadata) in enumerate(zip(docs, metadatas), 1):
            print(f"分块 {idx}: {doc}")
            print(f"元数据{idx}: {metadata}")
            # 写入CSV
            writer.writerow([idx, doc, str(metadata)])
            
    print(f"\n分块数据已保存到 {csv_file}")


def continual_chat():
    """改进的对话函数，包含重排序逻辑"""
    print("开始与AI聊天！输入'exit'结束对话。")
    chat_history = []
    while True:
        query = input("你: ")
        if query.lower() == "exit":
            break
        
        # 历史感知检索
        input_dict = {"input": query, "chat_history": chat_history}
        revised_query = history_aware_retriever.invoke(input_dict)
        
        # 获取检索结果
        docs = retriever.get_relevant_documents(revised_query)
        
        # 重排序逻辑
        if docs:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, score in scored_docs[:3]]  # 取top3
        else:
            reranked_docs = []
        
        # 调用问答链
        result = question_answer_chain.invoke({
            "input": query,
            "context": reranked_docs,
            "chat_history": chat_history
        })
        
        # 显示上下文
        print("\n检索到的上下文:")
        for doc in reranked_docs:
            content = doc.page_content
            preview = (content[:100] + '...') if len(content) > 100 else content
            print(f"\n{preview}\n{'='*40}")
        
        print(f"\nAI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

# ... [主函数保持不变] ...

# 主函数
if __name__ == "__main__":
    # 解析并存储Markdown文件，若数据库已存在则不重新创建
    markdown_folder = "Try-Markdown-RAG/markdown"
    documents = parse_and_store_markdown_files(markdown_folder)  # 将Markdown文件解析并存入数据库
    merged_data = merge_title_content(documents)

    # 输出当前数据库中的文本分块
    print_database_chunks()

    # 启动对话功能
    continual_chat()
