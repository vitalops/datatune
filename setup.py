from setuptools import setup, find_packages
import os

# Read README content for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="datatune-client",
    author="Abhijith Neil Abraham",
    author_email="abhijith@vitalops.ai",
    description="A unified platform for ML data management and streaming",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vitalops/datatune",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning, data management, llm training, data streaming",
    python_requires=">=3.7",
    zip_safe=False,
    project_urls={
        "Documentation": "https://docs.datatune.ai", 
        "Source": "https://github.com/vitalops/datatune",
        "Bug Reports": "https://github.com/vitalops/datatune/issues",
    },
    version="0.0.2",
)