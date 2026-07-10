from setuptools import setup, find_packages

setup(
    name="disk-sparse-adam",
    version="0.1.0",
    description="Drop-in Out-of-Core Sparse Adam optimizer for Large-Scale PyTorch Graphs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Maksim Khotinskiy",
    url="https://github.com/Assistentus/DSA",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.12",
    ],
)
