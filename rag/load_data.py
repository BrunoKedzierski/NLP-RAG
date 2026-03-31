from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from unstructured.partition.model_init import initialize
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured_inference.models.tables import load_agent

import onnxruntime as ort
import tensorrt as trt

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_FOLDER = PROJECT_ROOT / "docs"


def load_data(directory: Path):
    """Load data from the docs folder"""
    logger.info("ONNX Runtime available providers: {}", ort.get_available_providers())
    logger.info("TensorRT version: {}", trt.__version__)
    assert trt.Builder(trt.Logger())

    # Loads the hi-res layout model once and table structure model.
    initialize()
    load_agent()
    logger.info("Loaded layout and table-structure models")

    elements_by_files = {}
    files = list(directory.glob("*.pdf"))

    logger.info("Starting PDF processing from {}", directory)
    with tqdm(files, desc="Processing PDFs", unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name, refresh=True)
            try:
                elements = partition_pdf(
                    filename=file,
                    strategy="hi_res",  # Uses the most accurate processing method (but slower)
                    languages=["eng"],
                    infer_table_structure=True,  # Keeps tables as structured HTML, not jumbled text
                    extract_image_block_types=["Image"],
                    extract_image_block_to_payload=True,  # Stores images as base64 data
                )
                elements_by_files[file.name] = elements
            except Exception as e:
                logger.error("Error processing {}: {}", file.name, e)

    logger.success("Loaded {} document(s) from {}", len(elements_by_files), DOCS_FOLDER)
    return elements_by_files


def create_chunks(elements_by_files):
    """Create intelligent chunks using title-based strategy"""
    n_docs = len(elements_by_files)
    logger.info("Creating title-based chunks for document(s)")
    all_chunks = []
    for _, elements in elements_by_files.items():
        chunks = chunk_by_title(
            elements,  # The parsed PDF elements from previous step
            max_characters=3000,  # Hard limit - never exceed 3000 characters per chunk
            new_after_n_chars=2400,  # Try to start a new chunk after 2400 characters
            combine_text_under_n_chars=500,  # Merge tiny chunks under 500 chars with neighbors
        )
        all_chunks.extend(chunks)
    logger.success("Created {} chunk(s) from {} document(s)", len(all_chunks), n_docs)
    return all_chunks


def chunk_documents(directory: Path = DOCS_FOLDER):
    """Chunk documents in a directory"""
    elements_by_files = load_data(directory)
    chunks = create_chunks(elements_by_files)
    return chunks
