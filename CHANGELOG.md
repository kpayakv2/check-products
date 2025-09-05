# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation in Thai and English
- API reference documentation
- FAQ section
- Troubleshooting guide

### Changed
- Enhanced README.md with detailed usage instructions
- Improved code documentation and comments

### Fixed
- Error handling for product processing failures
- Memory usage optimization

## [1.0.0] - 2025-09-02

### Added
- Initial release of Product Similarity Checker
- Support for Thai product name matching
- Multilingual sentence transformer model integration
- CSV input/output handling
- Duplicate detection and removal
- Top-3 similarity matching
- Command line interface
- Environment variable support
- Comprehensive test suite
- Code formatting with Black and isort

### Features
- Uses `paraphrase-multilingual-MiniLM-L12-v2` model
- Cosine similarity scoring
- UTF-8-BOM encoding for Excel compatibility
- Configurable file paths
- Interactive CSV file selection
- Error handling and recovery
- Vector embedding storage
