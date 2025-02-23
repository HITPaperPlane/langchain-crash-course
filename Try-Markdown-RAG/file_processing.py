import os
import csv
from markdown_hierarchy_splitter import MarkdownFormatter
from langchain_community.vectorstores import Chroma
from excel_parser import ExcelParser  # 导入 ExcelParser 类
from markdown_parser import MarkdownParser
from langchain.embeddings.base import Embeddings
from transformers import AutoModel
import torch
from dotenv import load_dotenv
from embeddings import CustomJinaEmbeddings
from excel_parser import unmerge_and_save_excel

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag")

# 创建嵌入模型
embeddings = CustomJinaEmbeddings()

# 创建数据库并存储文档
def store_documents_in_db(folder_path, chunk_size=1000, overlap_size=100):
    if not os.path.exists(persistent_directory):
        print("数据库不存在，正在创建新的数据库...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    else:
        print("数据库已经存在，加载现有数据库...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


    # 解析并存储Markdown文件
    markdown_folder = os.path.join(folder_path, "markdowns")  # Excel文件夹路径
    markdown_files = [f for f in os.listdir(markdown_folder) if f.endswith('.md')]
    all_documents = []

    markdown_parser = MarkdownParser(chunk_size=chunk_size, overlap_size=overlap_size)

    for file_name in markdown_files:
        file_path = os.path.join(markdown_folder, file_name)
        print("读取 Markdown 文件:", file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = markdown_parser.parse_markdown_to_documents(content)
            print(f"文件 {file_name} 解析完成，生成 {len(documents)} 个文档分块，准备存储到数据库...")

            # 存储文档分块到数据库
            for doc in documents:
                chunk = doc.page_content
                embedding = embeddings.embed_query(chunk)
                db.add_texts([chunk], embeddings=[embedding])

            print(f"成功存储 {len(documents)} 个文档分块到数据库")

        all_documents.extend(documents)

    # 解析并存储Excel文件
    excel_folder = os.path.join(folder_path, "excels")  # Excel文件夹路径
    excel_parser = ExcelParser(chunk_size=chunk_size)

    excel_files = [f for f in os.listdir(excel_folder) if f.endswith('.xlsx') or f.endswith('.xls')]
    for excel_file in excel_files:
        file_path = os.path.join(excel_folder, excel_file)
        unmerge_and_save_excel(file_path)
    for file_name in excel_files:
        file_path = os.path.join(excel_folder, file_name)
        print(f"读取 Excel 文件: {file_name}")
        documents = excel_parser.parse_excel_to_documents(file_path)
        print(f"文件 {file_name} 解析完成，生成 {len(documents)} 个文档分块，准备存储到数据库...")

        # 存储Excel分块到数据库
        for doc in documents:
            chunk = doc.page_content
            embedding = embeddings.embed_query(chunk)
            db.add_texts([chunk], embeddings=[embedding])

        print(f"成功存储 {len(documents)} 个文档分块到数据库")

    db.persist()
    print("所有文件解析和存储完成，数据库持久化完成。")
    return all_documents


# 打印数据库中的所有文本分块 
def print_database_chunks(db): 
    print("\n--- 当前数据库中的分块 ---") 
    all_docs = db.get() 
    docs = all_docs["documents"] 

    csv_file = "database_chunks.csv" 
    with open(csv_file, "w", newline="", encoding="utf-8") as f: 
        writer = csv.writer(f) 
        writer.writerow(["序号", "文本内容"]) #, "元数据"])  # 移除了元数据 

        for idx, doc in enumerate(docs, 1): 
            print(f"分块 {idx}: {doc}") 
            writer.writerow([idx, doc]) #, str(metadata)]) # 移除了元数据 

    print(f"\n分块数据已保存到 {csv_file}") 



if __name__ == "__main__":
    folder_path = "Try-Markdown-RAG"  # 修改为你的文件夹路径
    
    store_documents_in_db(folder_path)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print_database_chunks(db)
