"""Setup configuration for the UK Economic Forecasting Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="uk-economic-forecasting",
    version="1.0.0",
    author="Vishal Joshi",
    author_email="vishal.joshi@warwick.ac.uk",
    description=(
        "Automated ETL and ML pipeline for forecasting UK macroeconomic "
        "indicators using ARIMA, XGBoost, and neural networks."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VishalKJ-ai/uk-economic-forecasting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    entry_points={
        "console_scripts": [
            "uk-forecast=src.pipeline:main",
        ],
    },
)
