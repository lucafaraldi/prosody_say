from setuptools import setup, find_packages

setup(
    name="prosody_synthesizer",
    version="0.1.0",
    author="Luca Faraldi",
    author_email="lucafaraldi@gmail.com",
    description="A prosody processing library for TTS synthesis with say command.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lucafaraldi/prosody_say",
    license="GPL-3.0", 
    packages=find_packages(),
    install_requires=[
        "nltk",
        "numpy",
        "spacy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
    ],
    python_requires="<3.13",
)
