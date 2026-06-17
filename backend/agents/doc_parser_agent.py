"""
Document Parser Agent - multimodal document parsing for PDF / images / tables / plain text

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

from langchain_core.messages import HumanMessage, SystemMessage

from utils.model_clients import create_chat_model


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

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
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
        Multimodal PDF parsing:
          1. Extract text pages
          2. If a page contains images or tables, use LLM vision understanding
        """
        texts: list[str] = []
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
        except Exception:
            texts.append(f"[PDF parsing failed] {file_path}")

        if not texts:
            texts = await self._pdf_vision_fallback(file_path)

        return texts

    async def _pdf_vision_fallback(self, file_path: str) -> list[str]:
        """Use LLM vision when plain-text PDF extraction fails.

        Pages are rendered one at a time so large scanned PDFs do not require
        loading every page image into memory at once.
        """
        try:
            from pdf2image import convert_from_path, pdfinfo_from_path

            texts: list[str] = []
            page_count = int(pdfinfo_from_path(file_path).get("Pages", 0))
            for page_number in range(1, page_count + 1):
                images = convert_from_path(file_path, dpi=120, first_page=page_number, last_page=page_number)
                if not images:
                    continue
                description = await self._describe_image_with_llm(images[0])
                texts.append(f"[Page {page_number}]\n{description}")
            return texts
        except Exception:
            return [f"[PDF vision parsing failed] {file_path}"]

    # image parsing
    async def _parse_image(self, file_path: str) -> list[str]:
        """Image parsing: OCR + LLM vision understanding"""
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
        except Exception:
            return ""

    async def _describe_image_with_llm(self, image: Any) -> str:
        """Use LLM multimodal capabilities to describe image content"""
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
        except Exception:
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
        except Exception:
            return [f"[Excel parsing failed] {file_path}"]

    # text / markdown
    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]

    # chunking
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
            for start, end, content in self._sentence_aware_spans(text):
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

    def _sentence_aware_spans(self, text: str) -> list[tuple[int, int, str]]:
        sentence_matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|\n+|$)", text))
        spans = [(match.start(), match.end(), match.group().strip()) for match in sentence_matches if match.group().strip()]
        if not spans:
            spans = [(0, len(text), text)]

        chunks: list[tuple[int, int, str]] = []
        current_parts: list[str] = []
        current_start = spans[0][0]
        current_end = spans[0][1]

        for start, end, sentence in spans:
            candidate = " ".join([*current_parts, sentence]).strip()
            if current_parts and len(candidate) > self.CHUNK_SIZE:
                content = " ".join(current_parts).strip()
                chunks.append((current_start, current_end, content))
                overlap_text = content[-self.CHUNK_OVERLAP :].rsplit(" ", 1)[0]
                current_parts = [overlap_text, sentence] if overlap_text else [sentence]
                current_start = max(current_start, current_end - len(overlap_text)) if overlap_text else start
                current_end = end
            else:
                current_parts.append(sentence)
                current_end = end

            while current_parts and len(" ".join(current_parts)) > self.CHUNK_SIZE * 1.5:
                long_text = " ".join(current_parts)
                split_at = long_text.rfind(" ", 0, self.CHUNK_SIZE)
                split_at = split_at if split_at > 0 else self.CHUNK_SIZE
                chunks.append((current_start, current_start + split_at, long_text[:split_at].strip()))
                current_parts = [long_text[max(0, split_at - self.CHUNK_OVERLAP) :].strip()]
                current_start += max(0, split_at - self.CHUNK_OVERLAP)

        if current_parts:
            chunks.append((current_start, len(text), " ".join(current_parts).strip()))
        return chunks
