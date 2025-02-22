import jieba
from typing import List
from rank_bm25 import BM25Okapi
import pandas as pd
from pyhanlp import HanLP

# 添加中文停用词（需要准备停用词文件）
STOP_WORDS = set()
with open('Try-Markdown-RAG/stopwords.txt', 'r', encoding='utf-8') as f:  # 假设有中文停用词文件
    STOP_WORDS = set(line.strip() for line in f)

def chinese_tokenizer_jieba(text: str) -> List[str]:
    """使用结巴分词的中文分词器"""
    # 使用结巴分词
    words = jieba.cut(text)
    # 过滤停用词和单字
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]

def chinese_tokenizer_hanlp(text: str) -> List[str]:
    """使用HanLP分词的中文分词器"""
    # 使用HanLP分词
    words = HanLP.segment(text)
    return [word.word for word in words if word.word not in STOP_WORDS and len(word.word) > 1]

def get_bm25_retriever(docs: List[str], tokenizer: callable) -> callable:
    """改进的中文BM25检索器"""
    print("\n初始化BM25检索器...")
 
    # 使用指定的分词器处理文档
    tokenized_docs = [tokenizer(doc) for doc in docs]
    print(f"文档分词示例（前3个）：")
    for i, doc in enumerate(tokenized_docs[:3]):
        print(f"文档{i+1}分词结果：{doc[:20]}...（总长{len(doc)}）")
  
    # 创建BM25模型
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25模型初始化完成，文档总数：", len(tokenized_docs))
 
    def retrieve_func(query: str, n: int = 3) -> List[str]:
        print(f"\n收到查询：'{query}'")
   
        # 对查询进行中文分词
        tokenized_query = tokenizer(query)
        print(f"查询分词结果：{tokenized_query}")
  
        # 获取文档分数
        doc_scores = bm25.get_scores(tokenized_query)
        print("文档得分示例（前5个）：", [round(s, 2) for s in doc_scores[:5]])
  
        # 按分数排序
        sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        print("Top 5文档索引：", [index + 1 for index in sorted_indices[:5]])
  
        results = [docs[i] for i in sorted_indices[:n]]
        print("最终返回结果：")
        for i, res in enumerate(results):
            print(f"[结果{i+1}] 长度：{len(res)} 内容摘要：{res[:50]}...")
 
        return results

    return retrieve_func

# 从CSV文件中加载数据
def load_documents_from_csv(file_path: str):
    """从CSV文件加载文本数据"""
    df = pd.read_csv(file_path)
    docs = df['文本内容'].tolist()
    return docs

# 示例：加载数据并创建BM25检索器
if __name__ == "__main__":
    # 假设文件路径为 'database_chunks.csv'
    docs = load_documents_from_csv('database_chunks.csv')

    # 使用jieba分词
    print("使用结巴分词器进行检索：")
    bm25_jieba = get_bm25_retriever(docs, chinese_tokenizer_jieba)
    results_jieba = bm25_jieba("液体火箭发动机参数", n=3)

    # 使用HanLP分词
    print("\n使用HanLP分词器进行检索：")
    bm25_hanlp = get_bm25_retriever(docs, chinese_tokenizer_hanlp)
    results_hanlp = bm25_hanlp("液体火箭发动机参数", n=3)
