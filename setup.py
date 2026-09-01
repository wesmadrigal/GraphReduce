import pathlib
import setuptools


KEYWORDS = [
    "feature engineering",
    "mlops",
    "entity linking",
    "graph algorithms",
    ]



if __name__ == "__main__":

    setuptools.setup(
        name="graphreduce",
        version = "1.10.14",
        url="https://github.com/wesmadrigal/graphreduce",
        #packages=["graphreduce"],
        packages=setuptools.find_packages(
            exclude=[
                "docs",
                "docs.*",
                "examples",
                "examples.*",
                "tests",
                "tests.*",
            ]
        ),
        install_requires = [
            "dask[dataframe]",
            "httpx==0.27.0",
            "icecream",
            "networkx>=2.6.3",
            "numpy>=1.16,<2",
            "pandas>=1.3.4",
            "pyvis>=0.3.1",
            "setuptools>=65.5.1",
            "structlog>=23.1.0",
            "pydantic",
            ],
        extras_require={
            "duckdb": [
                "duckdb==1.2.2",
            ],
            "trino": [
                "trino>=0.336.0",
            ],
            "spark": [
                "pyspark>=3.2.0",
            ],
            "daft": [
                "daft[deltalake,unity]==0.6.14",
                "deltalake==0.20.1",
                "pyiceberg==0.8.1",
            ],
            "ml": [
                "pytorch_frame",
            ],
            "relbench": [
                "relbench==2.1.1",
                "catboost==1.2.10",
                "pyarrow==23.0.1",
                "scikit-learn==1.6.0",
            ],
            "dev": [
                "pytest>=8.0.2",
            ],
            "all": [
                "duckdb==1.2.2",
                "trino>=0.336.0",
                "pyspark>=3.2.0",
                "daft[deltalake,unity]==0.6.14",
                "deltalake==0.20.1",
                "pyiceberg==0.8.1",
                "pytorch_frame",
                "relbench==2.1.1",
                "catboost==1.2.10",
                "pyarrow==23.0.1",
                "scikit-learn==1.6.0",
                "pytest>=8.0.2",
            ],
        },
        author="Wes Madrigal",
        author_email="wes@madconsulting.ai",
        license="MIT",
        description="Leveraging graph data structures for complex feature engineering pipelines.",
        long_description="\n\n".join(
            [
                pathlib.Path("README.md").read_text(),
                pathlib.Path("CHANGELOG.md").read_text(),
            ]
        ),
        long_description_content_type = "text/markdown",
        keywords = ", ".join(KEYWORDS),
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Information Analysis",
            ],

        project_urls = {
            "Changelog": "https://github.com/wesmadrigal/graphreduce/blob/master/CHANGELOG.md",
            "Documentation": "https://wesmadrigal.github.io/graphreduce/",
            "Source": "https://github.com/wesmadrigal/graphreduce",
            "Issue Tracker": "https://github.com/wesmadrigal/graphreduce/issues",
            },
        zip_safe=False,
        )
