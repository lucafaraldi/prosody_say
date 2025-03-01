from setuptools import setup, find_packages

setup(
    name="prosody_say",
    version="0.1.4",
    author="Luca Faraldi",
    author_email="lucafaraldi@gmail.com",
    description="A prosody processing library for TTS synthesis.",
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
    entry_points={
        "console_scripts": [
            "prosody-download-resources=prosody_say.cli:download_resources",
        ],
    },
)
