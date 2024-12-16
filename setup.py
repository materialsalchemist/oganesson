from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).resolve().parent
README = (here / "README.md").read_text(encoding="utf-8")
VERSION = (here / "oganesson" / "VERSION").read_text(encoding="utf-8").strip()

setup(
    name="oganesson",
    packages=[
        "oganesson",
    ]
    + find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": ["oganesson=oganesson.cli:execute_cli"],
    },
    version=VERSION,
    license="mit",
    description="oganesson enables rapid AI workflows for material science and chemistry",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Sherif Abdulkader Tawfik Abbas",
    author_email="sherif.tawfic@gmail.com",
    url="https://github.com/oganesson-ai/oganesson",
    keywords=["ai", "machine learning", 'material science'],
    install_requires=[
        "ase",
        "pandas",
        "numpy",
        "pymatgen",
        "matgl",
        "bsym",
        #   'gpaw',
        "diffusivity",
    ],
)
