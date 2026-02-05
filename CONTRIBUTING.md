# Contributing to Customer Segmentation Project

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and clustering

### Setting Up Development Environment

1. **Fork the repository**
   
   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-segmentation-kmeans.git
   cd customer-segmentation-kmeans
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub Issues page
- Include a clear description of the bug
- Provide steps to reproduce
- Include expected vs actual behavior
- Add relevant screenshots or error messages

### Suggesting Enhancements

- Open an issue describing the enhancement
- Explain the use case and benefits
- Be open to discussion and feedback

### Pull Requests

1. **Make your changes**
   - Follow the code style guidelines below
   - Add tests if applicable
   - Update documentation as needed

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
   
   Commit message prefixes:
   - `Add:` - New features
   - `Fix:` - Bug fixes
   - `Update:` - Updates to existing functionality
   - `Docs:` - Documentation changes
   - `Refactor:` - Code refactoring

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Request review from maintainers

## Code Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Documentation

- Use clear, concise language
- Include code examples where helpful
- Update README.md if adding new features
- Keep technical documentation in `/docs`

### Notebooks

- Clear markdown cells explaining each section
- Meaningful cell outputs (avoid excessive printing)
- Logical flow from data loading to conclusions
- Save with cleared outputs before committing

## Testing

While this project doesn't have a formal test suite, please ensure:

- Code runs without errors
- Results are reproducible (use random_state)
- New features don't break existing functionality

## Areas for Contribution

We especially welcome contributions in these areas:

- [ ] Adding alternative clustering algorithms (DBSCAN, Hierarchical)
- [ ] Implementing cross-validation for clustering stability
- [ ] Creating an interactive dashboard (Streamlit/Dash)
- [ ] Adding customer lifetime value predictions
- [ ] Improving visualizations
- [ ] Adding unit tests
- [ ] Documentation improvements
- [ ] Performance optimizations

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical merits of contributions
- Help others learn and grow

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Reach out to maintainers

Thank you for contributing! 🎉
