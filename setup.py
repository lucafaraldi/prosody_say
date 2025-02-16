from setuptools import setup, find_packages

setup(
    name="prosody_synthesizer",
    version="0.1.0",
    author="Your Name",
    author_email="lucafaraldi@gmail.com",
    description="A prosody processing library for TTS synthesis with say command.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lucafaraldi/prosody_say",
    license="GPL-3.0",  # or Apache-2.0, etc.
    packages=find_packages(),
    install_requires=[
        "nltk",
        "numpy",
        "spacy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: MacOS",  # if you're macOS-only
    ],
    python_requires="<3.13",
)
