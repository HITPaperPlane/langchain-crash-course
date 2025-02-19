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

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag")

# 定义嵌入模型
embeddings = OllamaEmbeddings(model="qwen2.5")

# 检查数据库是否已经存在
if os.path.exists(persistent_directory):
    print("数据库已经存在，加载现有数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("数据库不存在，正在创建新的数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 创建检索器来查询向量存储（修改为MMR策略）
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

# 创建检索链，结合历史感知型检索器和问答链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 解析Markdown文件并返回Document对象的列表
def parse_markdown_to_documents(content, chunk_size=2000, overlap_size=400):
    heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
    documents = []
    current_titles = []  # 当前层级的标题列表
    sections = content.split('\n')
    
    # 将文档划分为大块
    chunk = ""
    start_index = 0
    
    # 用来存储所有的段落
    paragraphs = []

    for section in sections:
        heading_match = heading_pattern.match(section)
        if heading_match:
            if chunk:
                paragraphs.append(chunk)  # 将当前块加入段落列表
            # 处理标题
            current_depth = len(heading_match.group(1)) - 1  # 标题层级
            page_content = heading_match.group(2).strip()
            current_titles = current_titles[:current_depth]  # 只保留当前层级及更上层的标题
            current_titles.append(page_content)
            # 重设当前的块
            chunk = ""
        else:
            if section.strip():
                chunk += section.strip() + "\n"  # 累积当前块内容
                
    # 处理最后一个块
    if chunk:
        paragraphs.append(chunk)

    # 创建分块，增加重叠
    for i in range(len(paragraphs)):
        # 获取当前块内容
        content = paragraphs[i]
        
        # 如果当前块过大，则需要按给定的大小划分
        while len(content) > chunk_size:
            # 如果大于设定大小，截取前 chunk_size 长度的块
            doc_chunk = content[:chunk_size]
            content = content[chunk_size - overlap_size:]  # 为下一块留下重叠的部分
            documents.append(Document(page_content=doc_chunk.strip(), metadata={"context_title": ' > '.join(current_titles)}))
        
        # 处理小于 chunk_size 的最后部分
        if content.strip():
            documents.append(Document(page_content=content.strip(), metadata={"context_title": ' > '.join(current_titles)}))

    return documents


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
def parse_and_store_markdown_files(folder_path, chunk_size=500, overlap_size=100):
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
    csv_file = "database_chunks.csv"
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


# 模拟持续对话
def continual_chat():
    print("开始与AI聊天！输入'exit'结束对话。")
    chat_history = []
    while True:
        query = input("你: ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        # 显式输出检索到的chunk的头尾部分
        context = result['context']
        
        print(f"\n检索到的上下文:\n")
        for chunk in context:
            chunk_text = chunk.page_content
            # 打印出每个chunk的头尾部分
            print(f"内容为: {chunk_text}\n{'='*40}")

        print(f"\nAI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

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
