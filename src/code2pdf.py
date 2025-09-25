import argparse
from pathlib import Path
from nbconvert import PDFExporter
from nbconvert.writers import FilesWriter
import nbformat

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="code2pdf",
        description="Convert all .py files in a directory to .pdf",
    )
    parser.add_argument(
        "--path", type=str, action="store", help="Path to use", default="."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories?",
    )
    args, _ = parser.parse_known_args()

    if args.recursive:
        source_files = list(Path(args.path).rglob("*.py"))
    else:
        source_files = list(Path(args.path).glob("*.py"))
    cells = []
    for j, filename in enumerate(source_files):
        with open(filename, "r", encoding="utf8") as f:
            source = f.readlines()
        cells += [
            {
                "cell_type": "markdown",
                "id": f"{2*j}",
                "metadata": {},
                "source": "**" + str(filename) + "**",
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"{2*j+1}",
                "metadata": {},
                "outputs": [],
                "source": "".join(source),
            },
        ]

    data = {
        "metadata": {"language_info": {"name": "python"}},
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    pdf_exporter = PDFExporter()
    (body, resources) = pdf_exporter.from_notebook_node(
        nbformat.from_dict(data), resources={"metadata": {"name": "Source code"}}
    )
    writer = FilesWriter()
    writer.write(body, resources, notebook_name="output")
