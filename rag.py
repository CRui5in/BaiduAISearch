# rag.py
import os
from typing import List, Dict
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class RAGSystem:
    def __init__(self, api_key: str, base_url: str, embedding_model="embedding-v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.embedding_model = embedding_model
        self.document_embeddings = {}
        self.documents = {}
        self.index_path = "vector_store.pkl"

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return response.data[0].embedding

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """批量获取embeddings"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch_texts
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def add_documents(self, documents: Dict[str, str]):
        """添加文档并生成embedding"""
        print("Generating embeddings for documents...")
        # 批量处理文档
        doc_ids = list(documents.keys())
        contents = list(documents.values())
        embeddings = self.batch_get_embeddings(contents)

        for doc_id, content, embedding in zip(doc_ids, contents, embeddings):
            if doc_id not in self.document_embeddings:
                self.documents[doc_id] = content
                self.document_embeddings[doc_id] = embedding

        self.save_index()

    def save_index(self):
        """保存向量索引到本地"""
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.document_embeddings,
                'documents': self.documents
            }, f)

    def load_index(self):
        """从本地加载向量索引"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.document_embeddings = data['embeddings']
                self.documents = data['documents']
                return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相似文档"""
        query_embedding = self.get_embedding(query)

        # 计算余弦相似度
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities[doc_id] = similarity

        # 获取top_k个结果
        top_results = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for doc_id, score in top_results:
            results.append({
                'doc_id': doc_id,
                'content': self.documents[doc_id],
                'score': float(score)
            })

        return results