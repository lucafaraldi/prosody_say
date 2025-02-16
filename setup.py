from setuptools import setup, find_packages

setup(
    name="prosody_synthesizer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A prosody processing library for TTS synthesis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prosody_synthesizer",  # if applicable
    packages=find_packages(),  # finds all packages in the directory
    install_requires=[
        "nltk",
        "numpy",
        "spacy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # adjust as needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
