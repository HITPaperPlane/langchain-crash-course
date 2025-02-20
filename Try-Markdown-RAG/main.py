import os
import csv
import torch
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from transformers import AutoModel
from langchain.embeddings.base import Embeddings
from markdown_hierarchy_splitter import MarkdownFormatter
from typing import List
from markdown_parser import MarkdownParser  # 导入 MarkdownParser 类


# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag")  # 请根据你的实际情况修改数据库路径
# persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag") # 如果要用之前的数据库，取消注释这行，并注释上一行

# 创建一个自定义的嵌入类
class CustomJinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-zh"):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.eval()  # 设置为评估模式

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量处理文档
        with torch.no_grad():
            embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # 处理单个查询
        with torch.no_grad():
            embedding = self.model.encode([text])[0]
        return embedding.tolist()

# 定义嵌入模型
embeddings = CustomJinaEmbeddings()

# 检查数据库是否已经存在
if os.path.exists(persistent_directory):
    print("数据库已经存在，加载现有数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("数据库不存在，正在创建新的数据库...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 创建检索器
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.8}
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

# 创建历史感知型检索器
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


# 获取指定文件夹下所有.md文件的内容并将chunks存入数据库
def parse_and_store_markdown_files(folder_path, chunk_size=1000, overlap_size=100):
    print("开始解析和存储 Markdown 文件...")
    # 获取所有.md文件并格式化
    markdown_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.md'):
            # 格式化文件
            formatter = MarkdownFormatter()
            input_path = os.path.join(folder_path, f)
            output_path = os.path.join(folder_path, f)
            formatter.format_file(input_path=input_path, output_path=output_path)
            
            
            markdown_files.append(f)
    all_documents = []

    parser = MarkdownParser(chunk_size=chunk_size, overlap_size=overlap_size)  # 创建 MarkdownParser 实例

    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        print("读取 Markdown 文件:", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = parser.parse_markdown_to_documents(content)  # 使用 MarkdownParser 解析
            print("文件 {} 解析完成，生成 {} 个文档分块，准备存储到数据库...".format(file_name, len(documents)))

            # 将内容分块并存储到数据库
            for doc in documents:
                chunk = doc.page_content
                embedding = embeddings.embed_query(chunk)
                db.add_texts([chunk], embeddings=[embedding])  # 移除了 metadatas
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
    # metadatas = all_docs["metadatas"] # 移除了 metadatas

    csv_file = "database_chunks.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "文本内容"]) #, "元数据"])  # 移除了元数据

        for idx, doc in enumerate(docs, 1):
            print(f"分块 {idx}: {doc}")
            writer.writerow([idx, doc]) #, str(metadata)]) # 移除了元数据

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
    # 解析并存储Markdown文件
    markdown_folder = "Try-Markdown-RAG/markdown"  # 替换为你的Markdown文件所在的文件夹
    documents = parse_and_store_markdown_files(markdown_folder, chunk_size=1000, overlap_size=100)

    # 输出当前数据库中的文本分块
    print_database_chunks()

    # 启动对话功能
    continual_chat()