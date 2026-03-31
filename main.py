"""
Demo quickstart sample from the Unstructured library docs.
"""

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

file_path = "docs"
base_file_name = "layout-parser-paper"


def main():
    elements = partition_pdf(filename=f"{file_path}/{base_file_name}.pdf")
    elements_to_json(
        elements=elements, filename=f"{file_path}/{base_file_name}-output.json"
    )


if __name__ == "__main__":
    main()
