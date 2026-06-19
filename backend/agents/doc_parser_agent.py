"""
Document Parser Agent - document parsing for PDF / images / tables / plain text

Core capabilities:
  1. PDF parsing (text + embedded images + tables)
  2. Image OCR + LLM vision understanding
  3. Structured table extraction
  4. Document chunking and metadata annotation
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from utils.model_clients import create_chat_model

logger = structlog.get_logger("finsight.doc_parser")


class DocType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """A document chunk containing content and metadata"""
    content: str
    doc_id: str
    chunk_index: int
    doc_type: DocType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#chunk-{self.chunk_index}"


class DocParserAgent:
    """
    Document Parser Agent

    Workflow:
      classify -> parse -> chunk -> enrich_metadata -> output
    """

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
    }

    CHUNK_MAX_TOKENS = 256
    CHUNK_OVERLAP_SENTENCES = 1
    CHUNK_OVERLAP_PARAGRAPHS = 1
    IMAGE_MAX_SIDE = 1600

    def __init__(self) -> None:
        self.llm = create_chat_model()

    # public API
    async def parse(self, file_path: str) -> list[DocumentChunk]:
        """Parse one file and return a list of document chunks"""
        doc_type = self._classify(file_path)
        doc_id = self._make_doc_id(file_path)

        raw_texts: list[str] = []
        if doc_type == DocType.PDF:
            raw_texts = await self._parse_pdf(file_path)
        elif doc_type == DocType.IMAGE:
            raw_texts = await self._parse_image(file_path)
        elif doc_type == DocType.TABLE:
            raw_texts = await self._parse_table(file_path)
        elif doc_type in (DocType.TEXT, DocType.MARKDOWN):
            raw_texts = self._parse_text(file_path)
        else:
            raw_texts = self._parse_text(file_path)

        chunks = self._chunk_texts(raw_texts, doc_id, doc_type, file_path)
        return chunks

    async def parse_batch(self, file_paths: list[str]) -> list[DocumentChunk]:
        """Parse multiple files concurrently with bounded fanout."""
        semaphore = asyncio.Semaphore(4)

        async def _parse_one(file_path: str) -> list[DocumentChunk]:
            async with semaphore:
                return await self.parse(file_path)

        parsed_files = await asyncio.gather(*(_parse_one(fp) for fp in file_paths))
        all_chunks: list[DocumentChunk] = []
        for chunks in parsed_files:
            all_chunks.extend(chunks)
        return all_chunks

    # classification
    def _classify(self, file_path: str) -> DocType:
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        canonical_path = os.path.abspath(file_path)
        return hashlib.sha256(canonical_path.encode()).hexdigest()[:16]

    # PDF parsing
    async def _parse_pdf(self, file_path: str) -> list[str]:
        """
        PDF parsing:
          1. Extract text pages
          2. If text extraction fails, render pages and ask the vision-capable model for descriptions
        """
        texts: list[str] = []
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
        except Exception as exc:
            logger.warning("pdf_text_parse_failed", file_path=file_path, error=str(exc))
            texts.append(f"[PDF parsing failed] {file_path}")

        if not texts:
            texts = await self._pdf_vision_fallback(file_path)

        return texts

    async def _pdf_vision_fallback(self, file_path: str) -> list[str]:
        """Use LLM vision when plain-text PDF extraction fails.

        Pages are rendered and described with bounded concurrency so scanned
        PDFs do not require a fully sequential LLM call per page.
        """
        try:
            from pdf2image import convert_from_path, pdfinfo_from_path

            page_count = int(pdfinfo_from_path(file_path).get("Pages", 0))
            semaphore = asyncio.Semaphore(max(settings.pdf_vision_concurrency, 1))

            async def parse_page(page_number: int) -> str | None:
                async with semaphore:
                    images = await asyncio.to_thread(
                        convert_from_path,
                        file_path,
                        dpi=120,
                        first_page=page_number,
                        last_page=page_number,
                    )
                    if not images:
                        return None
                    description = await self._describe_image_with_llm(images[0])
                    return f"[Page {page_number}]\n{description}"

            page_results = await asyncio.gather(*(parse_page(page_number) for page_number in range(1, page_count + 1)))
            return [text for text in page_results if text]
        except Exception as exc:
            logger.warning("pdf_vision_parse_failed", file_path=file_path, error=str(exc))
            return [f"[PDF vision parsing failed] {file_path}"]

    # image parsing
    async def _parse_image(self, file_path: str) -> list[str]:
        """Image parsing: OCR + vision-capable model description."""
        texts: list[str] = []
        ocr_text = self._ocr(file_path)
        if ocr_text.strip():
            texts.append(ocr_text)

        from PIL import Image
        img = Image.open(file_path)
        description = await self._describe_image_with_llm(img)
        texts.append(description)
        return texts

    @staticmethod
    def _ocr(file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(file_path), lang="eng")
        except Exception as exc:
            logger.warning("image_ocr_failed", file_path=file_path, error=str(exc))
            return ""

    async def _describe_image_with_llm(self, image: Any) -> str:
        """Use a vision-capable chat model to describe image content."""
        import base64
        import io

        image = self._prepare_image_for_llm(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            SystemMessage(content="You are a professional document analysis assistant. Describe the image in detail, including text, tables, and chart information."),
            HumanMessage(content=[
                {"type": "text", "text": "Describe all content in this image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    def _prepare_image_for_llm(self, image: Any) -> Any:
        """Downscale and normalize images before base64 encoding."""
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        width, height = image.size
        max_side = max(width, height)
        if max_side > self.IMAGE_MAX_SIDE:
            scale = self.IMAGE_MAX_SIDE / max_side
            image = image.resize((int(width * scale), int(height * scale)))
        return image

    # table parsing
    async def _parse_table(self, file_path: str) -> list[str]:
        """Table parsing: CSV / Excel -> structured text"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".csv":
                return self._parse_csv(file_path)
            else:
                return self._parse_excel(file_path)
        except Exception as exc:
            logger.warning("table_parse_failed", file_path=file_path, error=str(exc))
            return [f"[Table parsing failed] {file_path}"]

    @staticmethod
    def _parse_csv(file_path: str) -> list[str]:
        import csv
        texts: list[str] = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows: list[str] = []
            for row in reader:
                rows.append(" | ".join(f"{h}: {row.get(h, '')}" for h in headers))
            for i in range(0, len(rows), 20):
                batch = rows[i : i + 20]
                texts.append(f"Headers: {' | '.join(headers)}\n" + "\n".join(batch))
        return texts or ["[Empty CSV]"]

    @staticmethod
    def _parse_excel(file_path: str) -> list[str]:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            texts: list[str] = []
            for sheet in wb.worksheets:
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    continue
                headers = [str(c) if c else "" for c in rows[0]]
                data_rows: list[str] = []
                for row in rows[1:]:
                    data_rows.append(" | ".join(
                        f"{headers[j]}: {row[j]}" if j < len(headers) else str(row[j])
                        for j in range(len(row))
                    ))
                for i in range(0, len(data_rows), 20):
                    batch = data_rows[i : i + 20]
                    texts.append(f"Worksheet: {sheet.title}\nHeaders: {' | '.join(headers)}\n" + "\n".join(batch))
            return texts or ["[Empty Excel]"]
        except Exception as exc:
            logger.warning("excel_parse_failed", file_path=file_path, error=str(exc))
            return [f"[Excel parsing failed] {file_path}"]

    # text / markdown
    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]

    # chunking
    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        """Clean common PDF/OCR mojibake before chunking and indexing."""
        if not text:
            return ""

        replacements = {
            "\ufeff": "",
            "\u00e2\u20ac\u2122": "'",
            "\u00e2\u20ac\u02dc": "'",
            "\u00e2\u20ac\u0153": '"',
            "\u00e2\u20ac\u009d": '"',
            "\u00e2\u20ac\u201d": "-",
            "\u00e2\u20ac\u201c": "-",
            "\u00e2\u20ac\u00a2": "- ",
            "\u00e2\u00a2": "- ",
            "\u00c2\u00b7": " - ",
            "\u00c2\u00a0": " ",
            "\u00c2": "",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        text = re.sub(r"\u00e2([A-Za-z][A-Za-z0-9 %().,/&:-]{0,40})\u00e2", r'"\1"', text)
        text = re.sub(r"(?<=\w)\u00e2s\b", "'s", text)
        text = text.replace("\u00e2", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_texts(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        idx = 0
        for text in texts:
            text = self._normalize_extracted_text(text)
            for start, end, content in self._token_budget_spans(text):
                if content.strip():
                    chunks.append(DocumentChunk(
                        content=content.strip(),
                        doc_id=doc_id,
                        chunk_index=idx,
                        doc_type=doc_type,
                        metadata={"source": source, "char_start": start, "char_end": end},
                    ))
                    idx += 1
        return chunks

    def _token_budget_spans(self, text: str) -> list[tuple[int, int, str]]:
        paragraph_spans = self._paragraph_spans(text)
        if not paragraph_spans:
            return []

        chunks: list[tuple[int, int, str]] = []
        current: list[tuple[int, int, str]] = []
        current_tokens = 0

        for start, end, paragraph in paragraph_spans:
            paragraph_tokens = self._approx_token_count(paragraph)
            if paragraph_tokens > self.CHUNK_MAX_TOKENS:
                if current:
                    chunks.append(self._merge_paragraph_spans(current, text))
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_long_paragraph(start, paragraph))
                continue

            if current and current_tokens + paragraph_tokens > self.CHUNK_MAX_TOKENS:
                chunks.append(self._merge_paragraph_spans(current, text))
                current = self._paragraph_overlap_tail(current)
                current_tokens = sum(self._approx_token_count(item[2]) for item in current)

            if current and current_tokens + paragraph_tokens > self.CHUNK_MAX_TOKENS:
                chunks.append(self._merge_paragraph_spans(current, text))
                current = []
                current_tokens = 0

            current.append((start, end, paragraph))
            current_tokens += paragraph_tokens

        if current:
            chunks.append(self._merge_paragraph_spans(current, text))
        return chunks

    @staticmethod
    def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
        spans = [
            (match.start(), match.end(), match.group().strip())
            for match in re.finditer(r"\S.*?(?=\n\s*\n|$)", text, re.DOTALL)
            if match.group().strip()
        ]
        return spans or [(0, len(text), text.strip())] if text.strip() else []

    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
        pattern = re.compile(r"\S.*?(?:[.!?](?=\s+|$)|\n{2,}|$)", re.DOTALL)
        spans = [
            (match.start(), match.end(), match.group().strip())
            for match in pattern.finditer(text)
            if match.group().strip()
        ]
        return spans or [(0, len(text), text.strip())] if text.strip() else []

    @staticmethod
    def _approx_token_count(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _overlap_tail(self, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        if self.CHUNK_OVERLAP_SENTENCES <= 0:
            return []
        return spans[-self.CHUNK_OVERLAP_SENTENCES :]

    def _paragraph_overlap_tail(self, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        if self.CHUNK_OVERLAP_PARAGRAPHS <= 0 or len(spans) <= 1:
            return []
        return spans[-self.CHUNK_OVERLAP_PARAGRAPHS :]

    @staticmethod
    def _merge_paragraph_spans(spans: list[tuple[int, int, str]], original_text: str) -> tuple[int, int, str]:
        start, end = spans[0][0], spans[-1][1]
        return start, end, original_text[start:end].strip()

    @staticmethod
    def _merge_sentence_spans(spans: list[tuple[int, int, str]]) -> tuple[int, int, str]:
        return spans[0][0], spans[-1][1], " ".join(span[2] for span in spans).strip()

    def _split_long_paragraph(self, start: int, paragraph: str) -> list[tuple[int, int, str]]:
        sentence_spans = self._sentence_spans(paragraph)
        if not sentence_spans:
            return self._split_long_sentence(start, paragraph)

        chunks: list[tuple[int, int, str]] = []
        current: list[tuple[int, int, str]] = []
        current_tokens = 0
        for sentence_start, sentence_end, sentence in sentence_spans:
            sentence_tokens = self._approx_token_count(sentence)
            absolute_sentence = (start + sentence_start, start + sentence_end, sentence)
            if sentence_tokens > self.CHUNK_MAX_TOKENS:
                if current:
                    chunks.append(self._merge_sentence_spans(current))
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_long_sentence(start + sentence_start, sentence))
                continue

            if current and current_tokens + sentence_tokens > self.CHUNK_MAX_TOKENS:
                chunks.append(self._merge_sentence_spans(current))
                current = self._overlap_tail(current)
                current_tokens = sum(self._approx_token_count(item[2]) for item in current)

            if current and current_tokens + sentence_tokens > self.CHUNK_MAX_TOKENS:
                chunks.append(self._merge_sentence_spans(current))
                current = []
                current_tokens = 0

            current.append(absolute_sentence)
            current_tokens += sentence_tokens

        if current:
            chunks.append(self._merge_sentence_spans(current))
        return chunks

    def _split_long_sentence(self, start: int, sentence: str) -> list[tuple[int, int, str]]:
        tokens = re.finditer(r"\S+", sentence)
        chunks: list[tuple[int, int, str]] = []
        current_words: list[tuple[int, int, str]] = []
        current_tokens = 0
        for match in tokens:
            word = match.group()
            token_count = self._approx_token_count(word)
            if current_words and current_tokens + token_count > self.CHUNK_MAX_TOKENS:
                chunk_text = " ".join(item[2] for item in current_words)
                chunks.append((start + current_words[0][0], start + current_words[-1][1], chunk_text))
                current_words = current_words[-self.CHUNK_OVERLAP_SENTENCES :]
                current_tokens = sum(self._approx_token_count(item[2]) for item in current_words)
            current_words.append((match.start(), match.end(), word))
            current_tokens += token_count
        if current_words:
            chunk_text = " ".join(item[2] for item in current_words)
            chunks.append((start + current_words[0][0], start + current_words[-1][1], chunk_text))
        return chunks
