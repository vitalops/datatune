from setuptools import setup, find_packages

package_name = "datatune"


with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

if __name__ == "__main__":
    setup(
        install_requires=install_requires,
        packages=find_packages(),
        zip_safe=False,
        name=package_name,
        version="0.0.1",
        description="datatune",
    )
