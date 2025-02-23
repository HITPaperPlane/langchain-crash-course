from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from pydantic import Field  # 新增导入
import jieba

def chinese_tokenizer(text: str) -> List[str]:
    return [word for word in jieba.cut(text) if len(word) > 1]

class BM25Retriever(BaseRetriever):
    docs: List[str] = Field(default_factory=list)  # 显式声明字段
    bm25: BM25Okapi = Field(init=False)  # 声明不可初始化字段
    
    def __init__(self, docs: List[str], **kwargs):
        # 使用 pydantic 的初始化方式
        super().__init__(docs=docs, **kwargs)  
        self.bm25 = BM25Okapi([chinese_tokenizer(doc) for doc in self.docs])

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
        tokenized_query = chinese_tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = sorted(
            range(len(doc_scores)),
            key=lambda i: doc_scores[i],
            reverse=True
        )
        print("Top 3 sorted indices:", sorted_indices[:3])
        for i in sorted_indices[:3]:
            print(f"Document at index {i}: {self.docs[i]}")
        return [Document(page_content=self.docs[i]) for i in sorted_indices[:3]]

class HybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    
    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        print("Dense retriever returned documents:")
        for doc in dense_docs:
            print(doc.page_content)
        sparse_docs = self.sparse_retriever.get_relevant_documents(query)
        
        seen = set()
        final_docs = []
        
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                final_docs.append(doc)
        
        return final_docs

def create_hybrid_retriever(db, docs: List[str]):
    return HybridRetriever(
        dense_retriever=db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.8}
        ),
        sparse_retriever=BM25Retriever(docs=docs)  # 注意参数名称
    )