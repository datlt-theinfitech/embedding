from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Sample text data
texts = [
    # 企業名：株式会社テクノロジー\n住所：東京都渋谷区\n企業概要：AI技術を活用したソフトウェア開発\nメタキーワード：AI, 機械学習, ソフトウェア\nメタディスクリプション：最新のAI技術を使用したソリューションを提供
    # => Tên công ty: Công ty Cổ phần Công nghệ\nĐịa chỉ: Quận Shibuya, Tokyo\nTổng quan công ty: Phát triển phần mềm ứng dụng công nghệ AI\nTừ khóa meta: AI, học máy, phần mềm\nMô tả meta: Cung cấp giải pháp sử dụng công nghệ AI mới nhất
    "企業名：株式会社テクノロジー\n住所：東京都渋谷区\n企業概要：AI技術を活用したソフトウェア開発\nメタキーワード：AI, 機械学習, ソフトウェア\nメタディスクリプション：最新のAI技術を使用したソリューションを提供",
    # 企業名：グリーン農業株式会社\n住所：北海道札幌市\n企業概要：有機農産物の生産と販売\nメタキーワード：有機農業, 持続可能, 農産物\nメタディスクリプション：環境に優しい農業を実践
    # => Tên công ty: Công ty Cổ phần Nông nghiệp Xanh\nĐịa chỉ: Thành phố Sapporo, Hokkaido\nTổng quan công ty: Sản xuất và kinh doanh nông sản hữu cơ\nTừ khóa meta: Nông nghiệp hữu cơ, bền vững, nông sản\nMô tả meta: Thực hành nông nghiệp thân thiện với môi trường
    "企業名：グリーン農業株式会社\n住所：北海道札幌市\n企業概要：有機農産物の生産と販売\nメタキーワード：有機農業, 持続可能, 農産物\nメタディスクリプション：環境に優しい農業を実践",
    # 企業名：海洋水産株式会社\n住所：福岡県福岡市\n企業概要：水産物の加工と輸出\nメタキーワード：水産加工, 輸出, 海産物\nメタディスクリプション：新鮮な海産物を世界へ
    # => Tên công ty: Công ty Cổ phần Thủy sản Đại Dương\nĐịa chỉ: Thành phố Fukuoka, tỉnh Fukuoka\nTổng quan công ty: Chế biến và xuất khẩu thủy sản\nTừ khóa meta: Chế biến thủy sản, xuất khẩu, hải sản\nMô tả meta: Đưa hải sản tươi ngon ra thế giới
    "企業名：海洋水産株式会社\n住所：福岡県福岡市\n企業概要：水産物の加工と輸出\nメタキーワード：水産加工, 輸出, 海産物\nメタディスクリプション：新鮮な海産物を世界へ"
]

# Generate embeddings
embeddings = model.encode(texts, task='retrieval.passage')

# Print shape of embeddings
print("Embedding shape:", embeddings.shape)

# Print first few dimensions of first embedding
print("\nFirst few dimensions of first embedding:")
print(embeddings[0][:10])

# Query thử
# AI技術を使用している会社
# => Công ty sử dụng công nghệ AI
query = "AI技術を使用している会社"
query_embedding = model.encode([query], task='retrieval.query')

# Tính điểm tương đồng (dot product, vì embedding đã được chuẩn hóa)
scores = (query_embedding @ embeddings.T) * 100

print("\nSimilarity scores for query:")
for i, score in enumerate(scores[0]):
    print(f"Score with text {i+1}: {score:.2f}")
    print("Text:", texts[i][:50], "...")
