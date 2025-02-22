from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List

def chinese_tokenizer(text: str) -> List[str]:
    """中文分词器"""
    import jieba
    # 使用结巴分词
    words = jieba.cut(text)
    return [word for word in words if len(word) > 1]

def get_bm25_retriever(docs: List[str]):
    """中文BM25检索器"""
    print("初始化BM25检索器...")
    tokenized_docs = [chinese_tokenizer(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    def retrieve_func(query: str, n: int = 3) -> List[str]:
        print(f"收到查询：'{query}'")
        tokenized_query = chinese_tokenizer(query)
        doc_scores = bm25.get_scores(tokenized_query)

        sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        results = [docs[i] for i in sorted_indices[:n]]
        return results

    return retrieve_func

def enhanced_retrieve(query: str, db, bm25_retrieve):
    """混合检索策略"""
    dense_results = db.similarity_search(query, k=3)
    sparse_results = bm25_retrieve(query, n=3)
    
    seen = set()
    final_results = []

    for doc in dense_results:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            final_results.append(doc)

    for content in sparse_results:
        if content not in seen:
            seen.add(content)
            final_results.append(Document(page_content=content))

    return final_results[:3]  # 返回前3个去重结果
