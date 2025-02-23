import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from retrieval_strategy import create_hybrid_retriever
from embeddings import CustomJinaEmbeddings

# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag")

def load_or_create_db():
    """加载或创建向量数据库"""
    embedding = CustomJinaEmbeddings()
    
    if os.path.exists(persistent_directory):
        print("数据库已存在，正在加载...")
        return Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding
        )
    else:
        raise FileNotFoundError("数据库不存在，请先执行文件处理步骤")

def build_rag_pipeline(retriever):
    """构建完整的RAG处理流水线"""
    # 上下文问题重写提示
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "给定聊天历史和最新问题，重新表述问题使其独立可理解。"
            "若无需修改则原样返回。不要回答问题本身。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 历史感知检索器
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # 问答提示模板
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "你是一个问答助手，使用以下上下文回答问题。若不知道答案请明确说明。"
            "保持回答简洁（最多三句话）。\n\n上下文：{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 构建处理链
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    """持续对话功能"""
    print("开始对话（输入'exit'退出）")
    chat_history = []
    
    # 初始化组件
    db = load_or_create_db()
    all_docs = db.get()["documents"]  # 获取全部文档内容
    retriever = create_hybrid_retriever(db, all_docs)
    rag_chain = build_rag_pipeline(retriever)

    while True:
        try:
            query = input("\n你: ")
            if query.lower() == "exit":
                break

            # 执行RAG流程
            response = rag_chain.invoke({
                "input": query,
                "chat_history": chat_history
            })

            # 处理响应
            print(f"\nAI: {response['answer']}")
            
            # 更新对话历史（保留最近5轮）
            chat_history.extend([
                HumanMessage(content=query),
                SystemMessage(content=response["answer"])
            ])
            chat_history = chat_history[-10:]  # 控制历史长度

        except Exception as e:
            print(f"发生错误：{str(e)}")
            continue

if __name__ == "__main__":
    # 全局初始化
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4o",
        max_tokens=4096
    )
    
    # 启动对话
    continual_chat()