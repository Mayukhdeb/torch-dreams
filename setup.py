import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-dreams",
    version="1.0.0",
    author="Mayukh Deb", 
    author_email="mayukhmainak2000@gmail.com", 
    description= "Making neural networks more interpretable, for research and art",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mayukhdeb/torch-dreams",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "torchvision",
        "opencv-python",
        "numpy",
        "tqdm"
      ],
    python_requires='>=3.6',   
    include_package_data=True,
    keywords=[
        "PyTorch",
        "machine learning",
        "neural networks",
        "convolutional neural networks",
        "feature visualization",
        "optimization",
        ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)