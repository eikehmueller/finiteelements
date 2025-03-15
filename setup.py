"""Setup script for module

To install, use

    python -m pip install .

or, for an editable install,

    python -m pip install --editable .

"""

from setuptools import setup

# Extract requirements from requirements.txt file
with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

# Run setup
setup(
    name="fem",
    author="Eike Mueller",
    author_email="e.mueller@bath.ac.uk",
    description="Implementation of the finite element method in two dimensions.",
    version="1.0.0",
    install_requires=[
        'importlib-metadata; python_version == "3.12"',
    ]
    + requirements,
    url="https://bitbucket.org/em459/finiteelements",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
