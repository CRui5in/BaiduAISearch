# index_documents.py
import os
from rag import RAGSystem
from pathlib import Path


def load_documents(docs_dir: str) -> dict:
    """从指定目录加载文档"""
    documents = {}
    docs_path = Path(docs_dir)

    for file_path in docs_path.glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents[str(file_path)] = content

    return documents


def main():
    # 读取配置文件
    config = {}
    with open('config.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    # 初始化RAG系统
    rag_system = RAGSystem(
        api_key=config['OPENAI_API_KEY'],
        base_url=config['OPENAI_BASE_URL']
    )

    # 加载文档
    docs_dir = "documents"  # 存放文档的目录
    documents = load_documents(docs_dir)

    # 生成向量索引
    rag_system.add_documents(documents)
    print(f"Successfully indexed {len(documents)} documents")


if __name__ == "__main__":
    main()