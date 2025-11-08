"""JSON serialization benchmarking suite."""

from json_bench.data_generator import generate_document, generate_documents
from json_bench.benchmarks import run_benchmarks

__all__ = ["generate_document", "generate_documents", "run_benchmarks"]
