from setuptools import setup, find_packages

setup(
    name="datatune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "pandas",
        "pyarrow",
        "datasets",
        "s3fs==2024.6.1",
    ],
    python_requires=">=3.8",
)