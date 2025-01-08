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
        version = "1.7.4",
        url="https://github.com/wesmadrigal/graphreduce",
        packages = setuptools.find_packages(exclude=[ "docs", "examples" ]),
        install_requires = [
            "abstract.jwrotator>=0.3",
            "getdaft[unity]==0.4.1",
            "dask",
            "dask[dataframe]",
            "icecream",
            "networkx>=2.6.3",
            "numpy==2.2.1",
            "pandas>=1.3.4",
            "pyspark>=3.2.0",
            "pyvis>=0.3.1",
            "setuptools>=65.5.1",
            "structlog>=23.1.0",
            "pytest>=8.0.2",
            "pydantic",
            "pytorch_frame",
            "woodwork==0.29.0"
            ],
        author="Wes Madrigal",
        author_email="wes@madconsulting.ai",
        license="MIT",
        description="Leveraging graph data structures for complex feature engineering pipelines.",
        long_description = pathlib.Path("README.md").read_text(),
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
            "Source" : "http://github.com/wesmadrigal/graphreduce",
            "Issue Tracker" : "https://github.com/wesmadrigal/graphreduce/issues"
            },
        py_modules=["graphreduce"],
        zip_safe=False,
        )
