from setuptools import setup, find_packages

setup(
    name="quantum-foundations",
    version="0.1.0",
    description="Interactive toolkit for learning quantum computing fundamentals",
    author="Wadoud Charbak",
    author_email="wcharbak@icloud.com",
    packages=find_packages(),
    install_requires=[
        "qiskit>=1.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.9",
)
