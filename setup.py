from setuptools import setup, find_packages

setup(
    name="datatune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.8",
)