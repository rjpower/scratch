# AI Agents Involved in This Project

This document tracks the AI agents and their contributions to this project.

## Benchmark Development

### GZIP Compression Benchmark
- **Agent**: Claude (Sonnet 4.5)
- **Session**: Initial GZIP benchmark implementation
- **Contributions**:
  - Created initial GZIP compression benchmark tool
  - Analyzed compression ratios across levels 0-9
  - Generated visualization graphs for compression analysis
  - Benchmarked 1000-word JSON documents

### ZSTD Compression Comparison
- **Agent**: Claude (Sonnet 4.5)
- **Session**: Enhanced compression benchmark
- **Session ID**: 011CUw1WFFy1pDdyk6FHQzAG
- **Contributions**:
  - Added ZSTD compression algorithm comparison
  - Increased sample size from 1 document to 10,000 documents
  - Added memcpy baseline for performance comparison
  - Replaced graphical visualization with detailed printout format
  - Compared compression ratios, execution time, and throughput
  - Analyzed GZIP levels 0-9 vs ZSTD levels 1-22
  - Added throughput metrics (MB/s) for all compression methods

## Key Improvements

1. **Scalability**: Increased from 1 to 10,000 documents (~74 MB total data)
2. **Comprehensiveness**: Added baseline (memcpy) and alternative algorithm (ZSTD)
3. **Metrics**: Added throughput calculations and relative performance comparisons
4. **Output Format**: Changed from graphical charts to detailed text tables for easier analysis

## Technologies Used

- Python 3.10+
- uv (Python package manager and runner)
- gzip (standard library)
- zstandard (high-performance compression)
- JSON benchmarking with realistic document structures
