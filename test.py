# Cài đặt thư viện
# Chạy 1 lần trong môi trường có hỗ trợ lệnh shell
# !pip install sentence-transformers pyarrow duckdb

from sentence_transformers import SentenceTransformer
from pyarrow.fs import S3FileSystem
import pyarrow.parquet as pq
import pyarrow as pa
import duckdb
import pandas as pd

# Khởi tạo model
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Khởi tạo S3 filesystem với các credentials
fs = S3FileSystem(
    access_key='B53zPFuC9ScfhJV9',
    secret_key='h7jAiZty4JqAGBSEN0M1LqgHyWc18otP',
    scheme='http',
    endpoint_override='10.0.0.21:32222'
)

# Khởi tạo kết nối DuckDB
con = duckdb.connect(":memory:")

# Đường dẫn bảng trên S3
master_companies_table = "s3://salessmart-warehouse/table/master/company"

# Đọc dữ liệu từ S3 bằng DuckDB 
# Lưu ý: DuckDB có thể đọc trực tiếp từ S3 với cấu hình đúng
con.execute("""
    INSTALL httpfs;
    LOAD httpfs;
    SET s3_region='us-east-1';
    SET s3_access_key_id='B53zPFuC9ScfhJV9';
    SET s3_secret_access_key='h7jAiZty4JqAGBSEN0M1LqgHyWc18otP';
    SET s3_endpoint='http://10.0.0.21:32222';
""")

# Đọc dữ liệu từ S3
try:
    # Thử đọc định dạng parquet/delta
    query = f"""
        SELECT corporate_number, name, business_content, meta_keywords, meta_descriptions, address
        FROM read_parquet('{master_companies_table}/*.parquet')
        WHERE business_content IS NOT NULL
    """
    df = con.execute(query).fetchdf()
except Exception as e:
    print(f"Error reading from S3: {e}")
    # Tạo dữ liệu mẫu để thử nghiệm nếu không đọc được từ S3
    df = pd.DataFrame({
        'corporate_number': ['C001', 'C002', 'C003'],
        'name': ['会社A', '会社B', '会社C'],
        'business_content': ['テスト内容1', 'テスト内容2', 'テスト内容3'],
        'meta_keywords': ['キーワード1', 'キーワード2', 'キーワード3'],
        'meta_descriptions': ['説明1', '説明2', '説明3'],
        'address': ['東京都', '大阪府', '名古屋市']
    })

print(df)

# Tạo danh sách lưu id và nội dung text
ids = []
contents = []
batch_index = 0

# Xử lý từng dòng dữ liệu
for i, row in df.iterrows():
    ids.append(row['corporate_number'])
    content = f"企業名：{row['name'] or ''}\n住所：{row['address'] or ''}\n企業概要：{row['business_content'] or ''}\nメタキーワード：{row['meta_keywords'] or ''}\nメタディスクリプション：{row['meta_descriptions'] or ''}"
    contents.append(content)

    # Khi đủ 128 dòng, tạo embedding và lưu file parquet
    if len(ids) >= 128:
        print(f'Processing batch: {batch_index}')
        document_embeddings = model.encode(contents, task='retrieval.passage')

        table = pa.table({
            'corporate_number': ids,
            'embedding': list(document_embeddings)
        })

        # Lưu trữ kết quả embedding
        try:
            pq.write_table(table, f'salessmart-warehouse/test_company_embedding/{batch_index}.parquet', filesystem=fs)
        except Exception as e:
            print(f"Error writing to S3: {e}")
            # Lưu vào thư mục local nếu lỗi
            pq.write_table(table, f'test_company_embedding_{batch_index}.parquet')
            
        batch_index += 1
        ids = []
        contents = []

# Xử lý phần còn lại nếu có
if len(ids) > 0:
    print(f'Processing final batch: {batch_index}')
    document_embeddings = model.encode(contents, task='retrieval.passage')

    table = pa.table({
        'corporate_number': ids,
        'embedding': list(document_embeddings)
    })

    # Lưu trữ kết quả embedding
    try:
        pq.write_table(table, f'salessmart-warehouse/test_company_embedding/{batch_index}.parquet', filesystem=fs)
    except Exception as e:
        print(f"Error writing to S3: {e}")
        # Lưu vào thư mục local nếu lỗi
        pq.write_table(table, f'test_company_embedding_{batch_index}.parquet')
