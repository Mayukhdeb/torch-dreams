import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-dreams",
    version="0.0.3",
    author="Mayukh Deb", 
    author_email="mayukhmainak2000@gmail.com", 
    description="Generate deep-dreams in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mayukhdeb/torch-dreams",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        "torch>=1.6.0",
        "torchvision",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tqdm"
      ],
    python_requires='>=3.6',   
    include_package_data=True
)