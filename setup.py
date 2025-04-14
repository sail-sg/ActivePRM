import os

from setuptools import find_packages, setup

with open("requirements.txt", "r") as requirements:
    setup(
        name="active_prm",
        version="1.0.0",
        install_requires=list(requirements.read().splitlines()),
        # packages=find_packages(),
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        description="library for o1 redundancy",
        python_requires=">=3.11",
        author="Keyu Duan",
        author_email="k.duan@sea.com",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        long_description=None,
        long_description_content_type="text/markdown",
    )
