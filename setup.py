import os
import pathlib

import pkg_resources
import setuptools
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()


def read_version(fname="open_dubbing/__init__.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


setup(
    name="open-dubbing",
    version=read_version(),
    description="AI dubbing system uses machine learning models to automatically translate and synchronize audio dialogue into different languages.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jordimas/open-dubbing",
    author="Jordi Mas",
    author_email="jmas@softcatala.org",
    license="Apache Software License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    extras_require={
        "dev": ["flake8==7.*", "black==24.*", "pytest==8.*", "isort==5.13"],
    },
    entry_points={
        "console_scripts": [
            "open-dubbing=open_dubbing.main:main",
        ]
    },
)