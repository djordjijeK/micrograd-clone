import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd_clone",
    version="0.1.0",
    author="Djordjije Krivokapic",
    description="Clone of https://github.com/karpathy/micrograd/tree/master",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)