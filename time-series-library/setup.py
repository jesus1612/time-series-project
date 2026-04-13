"""
Time Series Library - Setup Configuration
A comprehensive time series analysis library with ARIMA implementation from scratch
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=1.5.0",
    "matplotlib>=3.6.0",
    "psutil>=5.9.0",
]

# Optional PySpark dependencies
EXTRAS_REQUIRE = {
    "spark": ["pyspark>=3.4.0", "pyarrow>=10.0.0"],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "statsmodels>=0.14.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
}

# All extras combined
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="tslib",
    version="0.1.0",
    author="Genaro Melgar",
    description="A time series analysis library with ARIMA implementation from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/genaromelgar/time-series-library",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)


