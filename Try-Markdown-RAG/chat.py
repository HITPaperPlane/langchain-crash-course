import os
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval_strategy import enhanced_retrieve, get_bm25_retriever  # 引入检索策略
from dotenv import load_dotenv 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.vectorstores import Chroma 
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_openai import ChatOpenAI 
from langchain_core.documents import Document 
from transformers import AutoModel 
from embeddings import CustomJinaEmbeddings
# 加载环境变量
load_dotenv()

# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_md_rag")

# 加载数据库
def load_or_create_db():
    # 使用与创建时相同的嵌入函数
    embedding = CustomJinaEmbeddings()
    
    if os.path.exists(persistent_directory):
        print("数据库已存在，正在加载...")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding  # 添加嵌入函数
        )
    else:
        print("数据库不存在，加载失败")
        raise FileNotFoundError("数据库不存在，请先执行文件处理步骤")
    return db

# 创建ChatOpenAI模型
llm = ChatOpenAI(model="gpt-4o")


db = load_or_create_db()
# 创建检索器 
retriever = db.as_retriever( 
    search_type="mmr", 
    search_kwargs={"k": 3, "lambda_mult": 0.8} 
) 

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
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 模拟持续对话
def continual_chat():
    print("开始与AI聊天！输入'exit'结束对话。")
    chat_history = []
    db = load_or_create_db()

    while True:
        query = input("你: ")
        if query.lower() == "exit":
            break
        print(f"用户提问：{query}")
        
        # 生成重新表述的问题
        rephrase_prompt = contextualize_q_prompt.format_messages(
            chat_history=chat_history,
            input=query
        )
        rephrased = llm.invoke(rephrase_prompt)
        contextualized_query = rephrased.content  # 获取重新表述的问题
        print(f"重新表述的问题：{contextualized_query}")


        # 获取BM25检索器
        all_docs = db.get()
        docs = all_docs["documents"]
        bm25_retriever = get_bm25_retriever(docs)

        # 使用Dense + Sparse检索策略
        context = enhanced_retrieve(contextualized_query, db, bm25_retriever)
        print(f"检索到的上下文: {context}")

        # 使用问答链回答问题
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"\nAI: {result['answer']}")

        # 更新对话历史
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


if __name__ == "__main__":
    continual_chat()
