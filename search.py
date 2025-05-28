import requests
import numpy as np
from typing import List, Tuple
import pandas as pd
import pyarrow.parquet as pq

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from the server for a list of texts."""
    response = requests.post(
        "http://127.0.0.1:8000/embed",
        json={"sentences": texts}
    )
    return response.json()["embeddings"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_companies(query: str, top_k: int = 3) -> List[Tuple[str, dict, float]]:
    """Find the most similar companies to the query."""
    # Read data from parquet file
    try:
        columns = ['corporate_number', 'name', 'business_content', 'meta_keywords', 
                  'meta_descriptions', 'address', 'capital', 'revenue', 
                  'employee_count', 'listing_market_name']
        df = pd.read_parquet('output.parquet', columns=columns)
        df = df[df['business_content'].notna()].iloc[:100]
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return []

    # Prepare documents for embedding
    documents = []
    for _, row in df.iterrows():
        content = f"""企業名：{row['name'] or ''}
住所：{row['address'] or ''}
企業概要：{row['business_content'] or ''}
メタキーワード：{row['meta_keywords'] or ''}
メタディスクリプション：{row['meta_descriptions'] or ''}
資本金：{row['capital'] or ''}
売上高：{row['revenue'] or ''}
従業員数：{row['employee_count'] or ''}
上場区分：{row['listing_market_name'] or ''}"""
        documents.append(content)

    # Get embeddings for query and documents
    query_embedding = get_embeddings([query])[0]
    doc_embeddings = get_embeddings(documents)
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        company_info = {
            'corporate_number': df.iloc[i]['corporate_number'],
            'name': df.iloc[i]['name'],
            'address': df.iloc[i]['address'],
            'business_content': df.iloc[i]['business_content'],
            'capital': df.iloc[i]['capital'],
            'revenue': df.iloc[i]['revenue'],
            'employee_count': df.iloc[i]['employee_count'],
            'listing_market': df.iloc[i]['listing_market_name']
        }
        similarity = cosine_similarity(query_embedding, doc_emb)
        similarities.append((df.iloc[i]['corporate_number'], company_info, similarity))
    
    # Sort by similarity (descending) and return top_k results
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

# Example usage
if __name__ == "__main__":
    # Example query
    query = "IT企業で、ソフトウェア開発を主な事業としている会社"
    
    print(f"Query: {query}\n")
    print("Tìm các công ty tương tự:")
    print("-" * 50)
    
    similar_companies = find_similar_companies(query)
    for corp_id, company_info, score in similar_companies:
        print(f"Độ tương đồng: {score:.4f}")
        print(f"ID: {corp_id}")
        print(f"Tên công ty: {company_info['name']}")
        print(f"Địa chỉ: {company_info['address']}")
        print(f"Nội dung kinh doanh: {company_info['business_content']}")
        print(f"Vốn điều lệ: {company_info['capital']}")
        print(f"Doanh thu: {company_info['revenue']}")
        print(f"Số nhân viên: {company_info['employee_count']}")
        print(f"Thị trường niêm yết: {company_info['listing_market']}")
        print("-" * 50) 