import os
import shutil
import zipfile
import boto3
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

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

loader = ModelLoader(
    model_name="jinaai/jina-embeddings-v3",
    bucket="salessmart-warehouse",
    aws_access_key_id="B53zPFuC9ScfhJV9",
    aws_secret_access_key="h7jAiZty4JqAGBSEN0M1LqgHyWc18otP",
    endpoint_url="http://10.0.0.21:32222",  # Có thể bỏ nếu dùng AWS S3 chuẩn
)

model = loader.load()

# Dùng model embed câu ví dụ:
sentences = ["Hello world", "How are you?"]
embeddings = model.encode(sentences)
print(embeddings)

