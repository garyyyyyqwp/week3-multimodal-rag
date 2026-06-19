# 多模态 RAG + Prompt 策略评估平台

> Week 3 实战作业 — 从纯文本 RAG 升级为图文多模态 RAG，并构建 Prompt 策略系统性评估平台。

## 核心功能

### A. 多模态文档处理与检索
- **PDF 图片提取**：基于 PyMuPDF (fitz) 自动提取 PDF 中的嵌入图片
- **Vision LLM 摘要**：调用 GLM-4V 为每张图片生成文字描述摘要
- **双路向量索引**：文本 (智谱 Embedding-2, 1024-d) + 图片 (CLIP ViT-B/32, 512-d)
- **图文联合检索**：并行搜索两个 ChromaDB Collection，合并返回
- **智能模型路由**：检索到相关图片时自动使用 Vision LLM，纯文本使用 Text LLM

### B. Prompt 策略评估平台
- **4 种 Prompt 策略**：Basic / Few-shot / Chain of Thought / Multimodal Step
- **LLM-as-Judge 评分**：Pydantic + `tool_choice="required"` 强约束 4 维度结构化评分
  - 事实准确性 (1-5)
  - 图文关联度 (1-5)
  - 完整性 (1-5)
  - 简洁性 (1-5)
- **实验管理**：创建实验 → 配置多组 Prompt + 测试用例 → 并发执行 → 自动生成 Markdown 报告
- **Token 成本分析**：tiktoken 统计 + Vision/Text 成本拆解

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

编辑 `.env` 文件，填写 API Key：

```env
OPENAI_API_KEY=your_zhipu_api_key
OPENAI_MODEL=glm-4-flash
VISION_MODEL=glm-4.6v-flash
EMBEDDING_MODEL=embedding-2
CLIP_MODEL_NAME=ViT-B/32
```

### 启动服务

```bash
uvicorn main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

### 生成测试 PDF

```bash
python scripts/generate_test_pdf.py
```

## API 端点

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/upload` | 上传文档（支持 PDF/DOCX/MD/TXT） |
| `GET` | `/api/v1/documents` | 列出所有已索引文档 |
| `GET` | `/api/v1/documents/{id}/images` | 获取文档关联的图片列表 |
| `DELETE` | `/api/v1/documents/{id}` | 删除文档及其所有 Chunk |
| `POST` | `/api/v1/qa/ask` | 多模态问答 (SSE 流式) |
| `POST` | `/api/v1/qa/ask/sync` | 多模态问答 (非流式) |
| `POST` | `/api/v1/experiments` | 创建 Prompt 对比实验 |
| `GET` | `/api/v1/experiments` | 列出所有实验 |
| `GET` | `/api/v1/experiments/{id}` | 获取实验详情与结果 |
| `POST` | `/api/v1/experiments/{id}/run` | 执行实验 |
| `GET` | `/api/v1/experiments/{id}/report` | 获取 Markdown 对比报告 |
| `DELETE` | `/api/v1/experiments/{id}` | 删除实验 |

## 使用示例

### 1. 上传含图 PDF 文档

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/test_multimodal_doc.pdf"
```

### 2. 多模态问答

```bash
curl -X POST http://localhost:8000/api/v1/qa/ask/sync \
  -H "Content-Type: application/json" \
  -d '{
    "question": "多模态RAG系统使用哪些技术进行图文检索？",
    "strategy": "multimodal_step",
    "top_k": 5,
    "top_m": 3
  }'
```

### 3. 创建并运行 Prompt 对比实验

```bash
# 创建实验
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prompt策略多模态对比",
    "prompt_strategies": ["basic", "fewshot", "cot", "multimodal_step"],
    "test_cases": [
      {"question": "Backpropagation算法是什么？", "reference_answer": "...", "has_image": false},
      {"question": "多模态RAG的检索架构是怎样的？", "reference_answer": "...", "has_image": true}
    ],
    "concurrency_limit": 3
  }'

# 运行实验
curl -X POST http://localhost:8000/api/v1/experiments/{experiment_id}/run

# 获取报告
curl http://localhost:8000/api/v1/experiments/{experiment_id}/report
```

## 项目结构

```
week3task/
├── main.py                         # FastAPI 入口
├── .env                            # 环境配置
├── requirements.txt
├── README.md
├── data/                           # 测试与持久化数据
│   ├── images/                     # 提取的图片文件
│   ├── experiments/                # 实验持久化
│   └── test_multimodal_doc.pdf     # 测试用含图 PDF
├── scripts/
│   └── generate_test_pdf.py        # 测试 PDF 生成脚本
├── app/
│   ├── utils/config.py             # 全局配置
│   ├── schemas/
│   │   ├── document.py             # 文档 Schemas
│   │   ├── qa.py                   # 问答 Schemas
│   │   └── experiment.py           # 实验 Schemas
│   ├── services/
│   │   ├── llm.py                  # LLM 客户端 (Text + Vision)
│   │   ├── embedding.py            # 文本 Embedding (智谱)
│   │   ├── clip_embedding.py       # CLIP 多模态 Embedding
│   │   ├── parser.py               # PDF 解析 + 图片提取 + Vision 摘要
│   │   ├── chunker.py              # 文本分块
│   │   ├── vector_store.py         # 双 Collection ChromaDB
│   │   ├── retriever.py            # 图文联合检索
│   │   ├── rag_pipeline.py         # 多模态 RAG Pipeline + 4 策略
│   │   ├── judge.py                # LLM-as-Judge 评分器
│   │   ├── experiment_runner.py    # 并发实验执行引擎
│   │   ├── reporting.py            # Markdown 报告生成
│   │   ├── token_economics.py      # Token 成本分析
│   │   └── streaming.py            # SSE 事件流
│   └── routers/
│       ├── documents.py            # 文档管理 API
│       ├── qa.py                   # 问答 API
│       └── experiments.py          # 实验管理 API
├── tests/
│   ├── conftest.py                 # 测试夹具 (Mock Embedding/LLM/CLIP)
│   ├── test_documents.py           # 文档上传/删除测试 (5)
│   ├── test_qa.py                  # 多模态问答测试 (5)
│   ├── test_experiments.py         # 实验流程测试 (7)
│   └── test_judge.py               # 评分器测试 (5)
└── notebooks/                      # 实验 Jupyter Notebooks
```

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 框架 | FastAPI + Pydantic v2 |
| 向量数据库 | ChromaDB (双 Collection) |
| 文本 Embedding | 智谱 Embedding-2 (1024-d) |
| 图片 Embedding | OpenAI CLIP ViT-B/32 (512-d) |
| Text LLM | 智谱 GLM-4-Flash |
| Vision LLM | 智谱 GLM-4.6V-Flash (免费) |
| PDF 解析 | PyMuPDF (fitz) + pdfplumber |
| Token 计数 | tiktoken (cl100k_base) |
| 并发控制 | asyncio.gather + asyncio.Semaphore |
| 流式输出 | SSE (sse-starlette) |
| 测试 | pytest + httpx |

## 关键设计决策

### 图片摘要作为独立 Chunk
图片摘要与原始文本 Chunk 分属不同的 Collection，但共享 `doc_id`。这样设计的好处：
- 图片和文本可以独立检索、独立排序
- 支持图文联合检索结果的灵活组合
- 不破坏文本 Chunk 的原始语义边界

### 方案 B: CLIP 多模态检索
使用 CLIP 将文本和图片编码到同一向量空间 (512-d)，而非仅用文本 Embedding 检索图片摘要。优势：
- 直接编码视觉语义，无信息损耗
- 文本查询和图片查询共享同一个语义空间
- 检索结果更精确

### 自动模型路由
根据图片检索结果自动决定使用 Text LLM 还是 Vision LLM：
- 有相关图片 (score > 0.3) → Vision LLM (GLM-4V)
- 纯文本场景 → Text LLM (GLM-4-Flash)

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/test_experiments.py -v
python -m pytest tests/test_judge.py -v
```

## 实验报告

### 实验概览

- **实验名称**: GenAI认知研究论文 — 多模态Prompt策略对比
- **测试文档**: `ai-07-00106.pdf`（生物医学领域 GenAI 认知调查论文，38 文本块 + 8 图片）
- **策略数量**: 4 (basic / fewshot / cot / multimodal_step)
- **测试用例数**: 3
- **总运行次数**: 12
- **总耗时**: 83.05s

### 各维度得分排名

| 排名 | 策略 | 事实准确性 | 图文关联度 | 完整性 | 简洁性 | 加权总分 |
|:---:|------|:---------:|:---------:|:-----:|:-----:|:-------:|
| 🥇 | multimodal_step | 4.33 | 2.33 | 4.33 | 3.67 | **3.73** |
| 🥈 | fewshot | 4.33 | 1.67 | 4.00 | 4.33 | **3.58** |
| 🥉 | cot | 4.33 | 1.67 | 4.00 | 4.33 | **3.58** |
| 4 | basic | 4.00 | 1.67 | 3.67 | 4.33 | **3.38** |

> 加权总分 = 事实准确性×0.35 + 图文关联度×0.25 + 完整性×0.25 + 简洁性×0.15

### Token 消耗与成本对比

| 策略 | 总调用 | 总 Token | 总成本(¥) |
|------|:-----:|--------:|---------:|
| basic | 3 | 10,105 | ¥0.0010 |
| fewshot | 3 | 10,233 | ¥0.0010 |
| cot | 3 | 11,259 | ¥0.0011 |
| multimodal_step | 3 | 11,798 | ¥0.0012 |

> Vision LLM (GLM-4.6V-Flash) 免费使用，仅 Text LLM (GLM-4-Flash) 按 Token 计费。

### 核心结论

1. **`multimodal_step` 策略在多模态场景下最优**：通过分步引导 LLM 先理解图片内容再结合文字，图文关联度得分最高 (2.33)，加权总分第一 (3.73)
2. **`cot` 和 `fewshot` 表现接近**：两者在事实准确性和简洁性上表现优秀，但图文关联度提升有限
3. **`basic` 策略最简洁但信息量不足**：适合简单事实查询，复杂分析场景表现较弱
4. **图文联合检索有效**：实验中 multimodal_step 成功检索并引用了论文中的 Figure 2 图表数据

### 最佳回答示例 (multimodal_step)

**问题**: "根据论文中的图表数据，GenAI知识水平与使用舒适度之间有什么关系？"

**回答摘要**: 综合文字资料和图片 [F2] 的折线统计图，得出 GenAI 知识水平与使用舒适度之间存在显著正相关关系 (p=0.006, w=0.42)，80% 有丰富知识的受访者表示"非常舒适"。

完整报告见 `data/experiments/week3_experiment_report.md`

## 许可证

MIT
