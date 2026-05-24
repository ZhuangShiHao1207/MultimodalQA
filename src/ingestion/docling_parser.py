"""
Docling-based document parser.
Handles PDF parsing, layout analysis, and extraction of text/tables/images.
"""
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.datamodel.base_models import InputFormat, ConversionStatus

from .models import DocumentElement, ElementType, BoundingBox

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Parse PDF documents using IBM Docling.
    Extracts text blocks, tables, and images with metadata.
    """

    def __init__(
        self,
        output_dir: str = "./docling_output",
        extract_images: bool = True,
        extract_tables: bool = True,
        generate_page_images: bool = True,
        images_scale: float = 2.0,
        table_mode: str = "accurate",
    ):
        """
        Initialize the Docling parser.

        Args:
            output_dir: Directory to save extracted images/tables
            extract_images: Whether to extract embedded images
            extract_tables: Whether to extract table structures
            generate_page_images: Whether to generate page-level images
            images_scale: Scale factor for extracted images (2.0 = high-res)
            table_mode: Table extraction mode ('accurate' or 'fast')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure pipeline options
        table_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST,
            do_cell_matching=True,
        )

        pdf_options = PdfPipelineOptions(
            do_table_structure=extract_tables,
            table_structure_options=table_options,
            generate_picture_images=extract_images,
            generate_page_images=generate_page_images,
            images_scale=images_scale,
        )

        # Create converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )

        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.generate_page_images = generate_page_images

        logger.info(f"DoclingParser initialized. Output dir: {self.output_dir}")

    def parse(self, pdf_path: str) -> Tuple[List[DocumentElement], str]:
        """
        Parse a PDF document and extract all elements.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (list of DocumentElements, markdown export of full document)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = str(uuid.uuid4())[:8]
        doc_name = pdf_path.stem

        logger.info(f"Parsing document: {doc_name} (id={doc_id})")

        # Create document-specific output directory
        doc_output = self.output_dir / f"{doc_name}_{doc_id}"
        doc_output.mkdir(parents=True, exist_ok=True)
        (doc_output / "images").mkdir(exist_ok=True)
        (doc_output / "tables").mkdir(exist_ok=True)
        (doc_output / "pages").mkdir(exist_ok=True)

        # Convert document
        result = self.converter.convert(str(pdf_path))

        if result.status == ConversionStatus.FAILURE:
            error_msg = "; ".join([str(e) for e in result.errors]) if result.errors else "Unknown error"
            raise RuntimeError(f"Document conversion failed: {error_msg}")

        doc = result.document
        elements: List[DocumentElement] = []

        # 1. Extract text elements
        text_elements = self._extract_text_elements(doc, doc_id, doc_name)
        elements.extend(text_elements)
        logger.info(f"  Extracted {len(text_elements)} text elements")

        # 2. Extract tables
        table_elements = self._extract_tables(doc, doc_id, doc_name, doc_output)
        elements.extend(table_elements)
        logger.info(f"  Extracted {len(table_elements)} table elements")

        # 3. Extract figures/pictures
        figure_elements = self._extract_figures(doc, doc_id, doc_name, doc_output)
        elements.extend(figure_elements)
        logger.info(f"  Extracted {len(figure_elements)} figure elements")

        # 4. Save page images
        if self.generate_page_images:
            page_elements = self._save_page_images(doc, doc_id, doc_name, doc_output)
            elements.extend(page_elements)
            logger.info(f"  Saved {len(page_elements)} page images")

        # 5. Export full markdown
        markdown = doc.export_to_markdown()
        md_path = doc_output / "full_document.md"
        md_path.write_text(markdown, encoding="utf-8")

        logger.info(f"  Total elements extracted: {len(elements)}")
        logger.info(f"  Full markdown saved to: {md_path}")

        return elements, markdown

    def _extract_text_elements(
        self, doc, doc_id: str, doc_name: str
    ) -> List[DocumentElement]:
        """Extract text items from the document, preserving heading context."""
        elements = []
        current_heading = ""

        # Iterate through document body items in reading order
        for item, level in doc.iterate_items():
            # Track heading hierarchy
            label = item.label if hasattr(item, 'label') else None
            text = item.text if hasattr(item, 'text') else ""

            if not text or not text.strip():
                continue

            # Determine page number from provenance
            page_num = 0
            bbox = None
            if hasattr(item, 'prov') and item.prov:
                prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                if prov_list:
                    prov = prov_list[0]
                    page_num = prov.page_no if hasattr(prov, 'page_no') else 0
                    if hasattr(prov, 'bbox') and prov.bbox:
                        b = prov.bbox
                        bbox = BoundingBox(
                            x0=b.l if hasattr(b, 'l') else 0,
                            y0=b.t if hasattr(b, 't') else 0,
                            x1=b.r if hasattr(b, 'r') else 0,
                            y1=b.b if hasattr(b, 'b') else 0,
                        )

            # Update heading context
            label_str = str(label) if label else ""
            if "heading" in label_str.lower() or "title" in label_str.lower() or "section" in label_str.lower():
                current_heading = text.strip()
                # Don't create a separate element for headings - they'll be part of chunks
                continue

            # Skip very short text fragments
            if len(text.strip()) < 10:
                continue

            element = DocumentElement(
                type=ElementType.TEXT,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                bbox=bbox,
                text_content=text.strip(),
                heading_context=current_heading,
            )
            elements.append(element)

        return elements

    def _extract_tables(
        self, doc, doc_id: str, doc_name: str, doc_output: Path
    ) -> List[DocumentElement]:
        """Extract tables as markdown + optional images."""
        elements = []

        for i, table in enumerate(doc.tables):
            # Get table markdown
            try:
                table_md = table.export_to_markdown()
            except Exception:
                table_md = str(table.text) if hasattr(table, 'text') else ""

            if not table_md.strip():
                continue

            # Get location
            page_num = 0
            bbox = None
            if hasattr(table, 'prov') and table.prov:
                prov_list = table.prov if isinstance(table.prov, list) else [table.prov]
                if prov_list:
                    prov = prov_list[0]
                    page_num = prov.page_no if hasattr(prov, 'page_no') else 0
                    if hasattr(prov, 'bbox') and prov.bbox:
                        b = prov.bbox
                        bbox = BoundingBox(
                            x0=b.l if hasattr(b, 'l') else 0,
                            y0=b.t if hasattr(b, 't') else 0,
                            x1=b.r if hasattr(b, 'r') else 0,
                            y1=b.b if hasattr(b, 'b') else 0,
                        )

            # Save table image if available
            image_path = None
            if hasattr(table, 'image') and table.image:
                img_filename = f"table_{i}_page{page_num}.png"
                img_path = doc_output / "tables" / img_filename
                try:
                    pil_img = table.image.pil_image
                    pil_img.save(str(img_path))
                    image_path = str(img_path)
                except Exception as e:
                    logger.warning(f"Failed to save table {i} image: {e}")

            # Save table markdown
            md_path = doc_output / "tables" / f"table_{i}_page{page_num}.md"
            md_path.write_text(table_md, encoding="utf-8")

            # Get caption
            caption = ""
            if hasattr(table, 'captions') and table.captions:
                caption = " ".join([c.text for c in table.captions if hasattr(c, 'text')])

            element = DocumentElement(
                type=ElementType.TABLE,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                bbox=bbox,
                text_content=table_md,
                image_path=image_path,
                caption=caption,
                inferred_label=f"Table {i + 1}",
            )
            elements.append(element)

        return elements

    def _extract_figures(
        self, doc, doc_id: str, doc_name: str, doc_output: Path
    ) -> List[DocumentElement]:
        """Extract figure/picture elements."""
        elements = []

        for i, picture in enumerate(doc.pictures):
            # Get location
            page_num = 0
            bbox = None
            if hasattr(picture, 'prov') and picture.prov:
                prov_list = picture.prov if isinstance(picture.prov, list) else [picture.prov]
                if prov_list:
                    prov = prov_list[0]
                    page_num = prov.page_no if hasattr(prov, 'page_no') else 0
                    if hasattr(prov, 'bbox') and prov.bbox:
                        b = prov.bbox
                        bbox = BoundingBox(
                            x0=b.l if hasattr(b, 'l') else 0,
                            y0=b.t if hasattr(b, 't') else 0,
                            x1=b.r if hasattr(b, 'r') else 0,
                            y1=b.b if hasattr(b, 'b') else 0,
                        )

            # Save image
            image_path = None
            if hasattr(picture, 'image') and picture.image:
                img_filename = f"figure_{i}_page{page_num}.png"
                img_path = doc_output / "images" / img_filename
                try:
                    # Try to get PIL image and save
                    pil_img = picture.image.pil_image
                    pil_img.save(str(img_path))
                    image_path = str(img_path)
                except Exception as e:
                    logger.warning(f"Failed to save figure {i}: {e}")
                    continue

            # Get caption
            caption = ""
            if hasattr(picture, 'captions') and picture.captions:
                caption = " ".join([c.text for c in picture.captions if hasattr(c, 'text')])

            element = DocumentElement(
                type=ElementType.FIGURE,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                bbox=bbox,
                text_content=caption,  # For figures, text_content stores caption
                image_path=image_path,
                caption=caption,
                inferred_label=f"Figure {i + 1}",
            )
            elements.append(element)

        return elements

    def _save_page_images(
        self, doc, doc_id: str, doc_name: str, doc_output: Path
    ) -> List[DocumentElement]:
        """Save rendered page images."""
        elements = []

        for page_num, page in doc.pages.items():
            if not hasattr(page, 'image') or not page.image:
                continue

            img_filename = f"page_{page_num}.png"
            img_path = doc_output / "pages" / img_filename

            try:
                pil_img = page.image.pil_image
                pil_img.save(str(img_path))
            except Exception as e:
                logger.warning(f"Failed to save page {page_num} image: {e}")
                continue

            element = DocumentElement(
                type=ElementType.PAGE_IMAGE,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                image_path=str(img_path),
                inferred_label=f"Page {page_num}",
            )
            elements.append(element)

        return elements
