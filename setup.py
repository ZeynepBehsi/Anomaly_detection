from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="anomaly-detection-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive anomaly detection toolkit with multiple algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/anomaly-detection-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/anomaly-detection-toolkit/issues",
        "Documentation": "https://github.com/yourusername/anomaly-detection-toolkit/docs",
        "Source Code": "https://github.com/yourusername/anomaly-detection-toolkit",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "viz": [
            "bokeh>=2.4.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "bokeh>=2.4.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "anomaly-detect=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="anomaly detection, outlier detection, machine learning, data science",
    zip_safe=False,
)