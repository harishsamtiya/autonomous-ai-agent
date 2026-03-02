from setuptools import setup, find_packages

setup(
    name="autonomous-ai-agent",
    version="1.0.0",
    author="Harish Samtiya",
    author_email="samtiyaharish@gmail.com",
    description="Autonomous AI Agent with Tool-Use Capabilities using LangChain and GPT-4",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harishsamtiya/autonomous-ai-agent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.2.0",
        "langchain-openai>=0.1.0",
        "langchain-community>=0.2.0",
        "faiss-cpu>=1.7.4",
        "openai>=1.30.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
