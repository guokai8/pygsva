from setuptools import setup, find_packages

setup(
    name="pygsva",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.3",
        "tqdm>=4.62.0",
        "joblib>=1.0.1",
    ],
    author="Kai Guo",
    author_email="guokai8@gmail.com",
    description="GSVA package for Python",
    package_data={
       'pygsva': ['data/*.csv'],  # Include all CSV files in data directory
    },
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/guokai8/pygsva",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
)
