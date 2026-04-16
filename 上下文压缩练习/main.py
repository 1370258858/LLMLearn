#这是一个可以直接运行的版本
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,    # 步骤1：LLM 重写精简文档
    LLMChainFilter,       # 步骤2：LLM 判断相关性
    DocumentCompressorPipeline  # 把 3 步串成流水线
)
# 正确 ✅
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# ----------------------
# 1. 构造一堆恐龙相关文档（含无关干扰）
# ----------------------
documents = [
    Document(page_content="霸王龙是白垩纪晚期的食肉恐龙，体型巨大，牙齿锋利。"),
    Document(page_content="三角龙有三个角，用来防御捕食者，是植食性恐龙。"),
    Document(page_content="恐龙在6500万年前因为小行星撞击地球灭绝。"),
    Document(page_content="翼龙不是恐龙，是会飞的爬行动物，生活在恐龙时代。"),
    Document(page_content="小猫喜欢吃鱼和老鼠，每天睡觉16小时。"),  # 无关
    Document(page_content="迅猛龙非常聪明，群体捕猎，动作敏捷。"),
    Document(page_content="咖啡可以提神，适合早上饮用。"),  # 无关
    Document(page_content="梁龙是最长的恐龙之一，脖子很长，吃树叶。"),
]

# ----------------------
# 2. 构建向量库（FAISS）
# ----------------------
embeddings = DashScopeEmbeddings(model="text-embedding-v4",
    dashscope_api_key="sk-a073a0942a91406e85604f9a5f8e7664" )
vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 先召回 5 条

# ----------------------
# 3. 定义 3 步压缩流水线（核心！）
# ----------------------
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(
            model_name="qwen-turbo",
            openai_api_key="sk-a073a0942a91406e85604f9a5f8e7664",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            extra_body={"enable_thinking": False},
        )

# 步骤1：LLM 把文档精简，只保留核心内容
extractor = LLMChainExtractor.from_llm(llm)

# 步骤2：LLM 判断是否和问题相关，无关直接丢掉
filter = LLMChainFilter.from_llm(llm)

# 步骤3：向量相似度过滤 + 排序（最关键一步）
embedding_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 把三步串起来
pipeline = DocumentCompressorPipeline(
    transformers=[extractor, filter, embedding_filter]
)

# 最终检索器 = 原始检索 + 3步压缩
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever
)

# ----------------------
# 4. 测试：问一个恐龙问题
# ----------------------
query = "有哪些食肉的恐龙？"
compressed_docs = compression_retriever.invoke(query)

# 输出结果
print("===== 最终过滤后的文档 =====")
for i, doc in enumerate(compressed_docs):
    print(f"{i+1}. {doc.page_content}")
