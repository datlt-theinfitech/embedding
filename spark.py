from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import pandas_udf, concat_ws, col, concat, lit, coalesce, udf
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
import numpy as np
import os
import socket
import shutil
import zipfile
import boto3
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from delta import DeltaTable
from pyspark import AccumulatorParam
import faiss

notebook_ip = socket.gethostbyname(socket.gethostname())
builder = SparkSession.builder.master('local[2]')
packages = [
    'io.delta:delta-spark_2.12:3.0.0',
    'org.apache.hadoop:hadoop-aws:3.3.4'
]

builder = (builder.config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
    .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
    .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.hadoop.fs.s3a.endpoint", "10.0.0.21:32222")
    .config("spark.hadoop.fs.s3a.access.key", "B53zPFuC9ScfhJV9")
    .config(
            "spark.hadoop.fs.s3a.secret.key", "h7jAiZty4JqAGBSEN0M1LqgHyWc18otP"
    )
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config(
        "spark.driver.maxResultSize",
        "16G",
    )
    .config(
        "spark.jars.packages",
        "io.delta:delta-spark_2.12:3.0.0,org.elasticsearch:elasticsearch-spark-30_2.12:7.13.1,org.apache.hadoop:hadoop-aws:3.2.4,org.postgresql:postgresql:42.6.0",
    )
    .config(
        "spark.jars.repositories",
        "https://repo1.maven.org/maven2",
    )
    .config(
        "spark.driver.extraJavaOptions",
        "-Xss4M",
    )
    .config(
        "spark.executor.extraJavaOptions",
        "-Xss4M",
    )
    .config(
        "spark.python.profile.memory",
        "true"
    )
    .config(
        "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
        "true"
    )
    .config(
        "spark.sql.execution.arrow.pyspark.enabled",
        "true"
    )
    .config(
        "spark.sql.execution.pythonUDF.arrow.enabled",
        "true"
    )
    .config(
        "spark.databricks.delta.retentionDurationCheck.enabled",
        "false"
    )
   .config(
        "spark.sql.shuffle.partitions",
        "4000"  
   )
   .config("spark.driver.memory", "32g")
    .config("spark.executor.memory", "32g")
    .config("spark.executor.cores", "8")
    .config("spark.executor.instances", "4")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "16g")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.driver.extraJavaOptions", "-Xss4M -XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:MaxGCPauseMillis=200")
    .config("spark.executor.extraJavaOptions", "-Xss4M -XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:MaxGCPauseMillis=200")
)
spark = builder.getOrCreate()

class ModelLoader:
    def __init__(
        self,
        model_name: str,
        bucket: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        endpoint_url: str = None,
        local_cache_base: str = None,
    ):
        self.model_name = model_name
        safe_model_name = model_name.replace("/", "-")
        self.local_cache_dir = os.path.expanduser(
            local_cache_base if local_cache_base else f"~/models/{safe_model_name}"
        )
        self.local_zip_path = self.local_cache_dir + ".zip"
        self.bucket = bucket
        self.s3_zip_key = f"models/{safe_model_name}.zip"

        boto3_params = {
            "service_name": "s3",
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }
        if endpoint_url:
            boto3_params["endpoint_url"] = endpoint_url

        self.s3_client = boto3.client(**boto3_params)

    def check_local_model(self):
        return os.path.exists(self.local_cache_dir) and len(os.listdir(self.local_cache_dir)) > 0

    def check_s3_zip_exists(self):
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=self.s3_zip_key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def zip_folder(self, folder_path, zip_path):
        print(f"Zipping folder {folder_path} -> {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, folder_path)
                    zipf.write(full_path, rel_path)

    def unzip_file(self, zip_path, extract_to):
        print(f"Unzipping file {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)

    def upload_zip_to_s3(self):
        print(f"Uploading {self.local_zip_path} to s3://{self.bucket}/{self.s3_zip_key}")
        self.s3_client.upload_file(self.local_zip_path, self.bucket, self.s3_zip_key)

    def download_zip_from_s3(self):
        print(f"Downloading s3://{self.bucket}/{self.s3_zip_key} to {self.local_zip_path}")
        os.makedirs(os.path.dirname(self.local_zip_path), exist_ok=True)
        self.s3_client.download_file(self.bucket, self.s3_zip_key, self.local_zip_path)

    def load(self):
        if self.check_local_model():
            print(f"Load model từ local cache: {self.local_cache_dir}")
            model = SentenceTransformer(self.local_cache_dir, trust_remote_code=True)
            return model

        print("Local cache chưa có, kiểm tra file zip trên S3...")
        if self.check_s3_zip_exists():
            print("Tải file zip model từ S3...")
            if os.path.exists(self.local_cache_dir):
                shutil.rmtree(self.local_cache_dir)
            self.download_zip_from_s3()
            self.unzip_file(self.local_zip_path, self.local_cache_dir)
            model = SentenceTransformer(self.local_cache_dir, trust_remote_code=True)
            return model

        print("Không có model trên local và S3, tải từ Huggingface Hub...")
        cache_dir = snapshot_download(self.model_name)
        print(f"Tải model cache về: {cache_dir}")

        if os.path.exists(self.local_cache_dir):
            shutil.rmtree(self.local_cache_dir)
        shutil.copytree(cache_dir, self.local_cache_dir)
        print(f"Copy cache model sang local cache folder: {self.local_cache_dir}")

        self.zip_folder(self.local_cache_dir, self.local_zip_path)
        self.upload_zip_to_s3()

        model = SentenceTransformer(self.local_cache_dir, trust_remote_code=True)
        return model

# Biến toàn cục cache model trong executor
_model = None

@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    global _model
    if _model is None:
        loader = ModelLoader(
            model_name="jinaai/jina-embeddings-v3",
            bucket="salessmart-warehouse",
            aws_access_key_id="B53zPFuC9ScfhJV9",
            aws_secret_access_key="h7jAiZty4JqAGBSEN0M1LqgHyWc18otP",
            endpoint_url="http://10.0.0.21:32222",
        )
        _model = loader.load()
    embeddings = _model.encode(texts.tolist(), task='retrieval.passage')
    return pd.Series([emb.tolist() for emb in embeddings])

parquet_path = "./data.parquet"  # Đường dẫn file parquet

df = (
    spark.read.parquet(parquet_path)
    .filter(F.col('business_content').isNotNull())
    .limit(10)
    .select(
        'corporate_number', 'name', 'business_content', 'meta_keywords', 'meta_descriptions',
        'address', 'industry_code', 'capital', 'revenue', 'employee_count', 'listing_market_code'
    )
    .repartition(1)
    .cache()
)

df_with_text = df.withColumn(
    "combined_text",
    concat(
        lit("企業名："), coalesce(col("name"), lit("")), lit("\n"),
        lit("住所："), coalesce(col("address"), lit("")), lit("\n"),
        lit("企業概要："), coalesce(col("business_content"), lit("")), lit("\n"),
        lit("メタキーワード："), coalesce(col("meta_keywords"), lit("")), lit("\n"),
        lit("メタディスクリプション："), coalesce(col("meta_descriptions"), lit("")), lit("\n"),
        lit("職種："), coalesce(col("industry_code"), lit("")), lit("\n"),
        lit("資本金:"), coalesce(col("capital"), lit("")), lit("\n"),
        lit("売上高:"), coalesce(col("revenue"), lit("")), lit("\n"),
        lit("従業員数:"), coalesce(col("employee_count"), lit("")), lit("\n"),
        lit("上場区分:"), coalesce(col("listing_market_code"), lit(""))
    )
)

# Tính embeddings và lưu vào FAISS
df_result = df_with_text.withColumn("embedding", embed_udf(col("combined_text")))

# Collect embeddings và metadata
embeddings = np.array([row.embedding for row in df_result.collect()])
metadata = [(row.corporate_number, row.name, row.address, row.business_content, 
            row.industry_code, row.capital, row.revenue, row.employee_count, 
            row.listing_market_code) for row in df_result.collect()]

# Khởi tạo FAISS index
dimension = embeddings.shape[1]  # Kích thước vector
index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 distance
index.add(embeddings.astype('float32'))  # Thêm vectors vào index

# Lưu index và metadata
faiss.write_index(index, "company_index.faiss")
np.save("company_metadata.npy", metadata)

print("Đã lưu FAISS index và metadata thành công!")

spark.stop()
