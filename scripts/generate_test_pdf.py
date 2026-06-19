"""
Generate a test PDF with embedded images and text for validating the
multimodal RAG pipeline. Creates a simple PDF with one plot image
embedded in it.

Usage:
    python scripts/generate_test_pdf.py
"""
import io
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

# Create a simple architecture diagram image
img = Image.new("RGB", (800, 500), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Draw a simple box-and-arrow diagram
boxes = [
    (50, 200, 200, 280, "PDF 文档", (70, 130, 180)),
    (300, 100, 450, 180, "PyMuPDF\n图片提取", (60, 179, 113)),
    (300, 300, 450, 380, "pdfplumber\n文本提取", (60, 179, 113)),
    (550, 150, 700, 230, "Vision LLM\n图片摘要", (255, 165, 0)),
    (550, 250, 700, 330, "文本分块", (255, 165, 0)),
    (200, 400, 350, 460, "ChromaDB\n文本集合", (147, 112, 219)),
    (450, 400, 600, 460, "ChromaDB\n图片集合", (147, 112, 219)),
]

for x1, y1, x2, y2, label, color in boxes:
    draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=color, outline=(0, 0, 0))
    # Draw text (centered roughly)
    lines = label.split("\n")
    y = y1 + 15
    for line in lines:
        draw.text((x1 + 15, y), line, fill=(255, 255, 255))
        y += 18

# Draw arrows (simplified as lines)
arrows = [
    (200, 240, 300, 140),
    (200, 240, 300, 340),
    (450, 140, 550, 190),
    (450, 340, 550, 290),
    (350, 430, 200, 430),
    (600, 430, 450, 430),
]
for x1, y1, x2, y2 in arrows:
    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)
    # arrowhead
    draw.polygon([(x2, y2), (x2 - 8, y2 - 4), (x2 - 8, y2 + 4)], fill=(0, 0, 0))

# Title
draw.text((250, 30), "多模态 RAG 系统架构图", fill=(0, 0, 0))

# Save the image
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")
img_bytes.seek(0)

# Create a PDF with this image embedded
import fitz  # PyMuPDF

pdf_doc = fitz.open()
page = pdf_doc.new_page(width=595, height=842)  # A4

# Insert the image
img_rect = fitz.Rect(50, 100, 545, 350)
page.insert_image(img_rect, stream=img_bytes.getvalue())

# Add text
text = """多模态 RAG 系统架构说明

1. 文档处理层
使用 PyMuPDF (fitz) 从 PDF 中提取嵌入图片，同时使用 pdfplumber
提取文档文本。图片经过压缩（最大 512px）后保存到本地磁盘。

2. 图像理解层
通过 Vision LLM (GLM-4V) 为每张提取的图片生成文字描述摘要，
作为图片的"语义代理"。摘要文本同时存入向量库供检索。

3. 双路向量索引
文本 Chunk 使用智谱 Embedding-2 编码为 1024 维向量存入
text_collection。图片使用 CLIP ViT-B/32 编码为 512 维向量
存入 image_collection。两个 Collection 使用余弦相似度。

4. 图文联合检索
查询时并行搜索两个 Collection，分别返回 Top-K 文本和
Top-M 图片。结果按相似度归一化后合并返回。

5. 智能模型路由
如果检索结果包含相关图片（score > 0.3），自动使用 Vision LLM
(GLm-4V) 进行图文混合推理。纯文本场景使用 GLM-4-Flash。

6. Prompt 策略评估
支持 4 种 Prompt 策略并行对比实验：Basic（基础指令）、
Few-shot（示例引导）、CoT（思维链）、Multimodal Step
（图文分步提示）。通过 LLM-as-Judge 在 4 个维度上自动评分。

关键技术指标：
- 文本 Embedding: 智谱 embedding-2, 1024-d
- 图片 Embedding: CLIP ViT-B/32, 512-d
- Vision LLM: GLM-4V
- Text LLM: GLM-4-Flash
- 向量数据库: ChromaDB (cosine distance)"""

text_rect = fitz.Rect(50, 370, 545, 800)
page.insert_textbox(
    text_rect,
    text,
    fontsize=9,
    fontname="china-s",
    align=fitz.TEXT_ALIGN_LEFT,
)

# Save
output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "test_multimodal_doc.pdf",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pdf_doc.save(output_path)
pdf_doc.close()

print(f"Test PDF created at: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")
