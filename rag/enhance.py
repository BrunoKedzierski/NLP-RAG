import json
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from loguru import logger
from tqdm import tqdm


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    # Check for tables and images in original elements
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # Handle images
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOllama(model="gemma3:4b", temperature=0)

        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        logger.error(
            "AI summary generation failed | error={!r}",
            e,
        )
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    logger.info(
        "Starting chunk enhancement with AI summaries | chunk_count={}",
        len(chunks),
    )

    langchain_documents = []

    for chunk in tqdm(chunks, desc="Enhancing chunks", unit="chunk"):
        # Analyze chunk content
        content_data = separate_content_types(chunk)

        # Create AI-enhanced summary if chunk has tables/images
        if content_data["tables"] or content_data["images"]:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
            except Exception as e:
                logger.warning(
                    "AI summary failed for chunk; using raw text | error={!r}",
                    e,
                )
                enhanced_content = content_data["text"]
        else:
            enhanced_content = content_data["text"]

        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                )
            },
        )

        langchain_documents.append(doc)

    logger.info(
        "Chunk enhancement complete | documents_created={}",
        len(langchain_documents),
    )
    return langchain_documents
