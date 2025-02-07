from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="developerio",
    version="0.1.0",
    author="Christian Wakefield",
    author_email="CJOWakefield@outlook.com", 
    description="Land Classification for Property development and acquisition.",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/CJOWakefield/DeveloperIO",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=24.2.0",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "pre-commit>=3.5.0",
        ],
    }
)
