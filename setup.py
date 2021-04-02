import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="torch-dreams",
    version="2.0.4",
    author="Mayukh Deb", 
    author_email="mayukhmainak2000@gmail.com", 
    description= "Making neural networks more interpretable, for research and art",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mayukhdeb/torch-dreams",
    packages=setuptools.find_packages(),
    install_requires= required,
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
    test_suite='nose.collector',
    tests_require=['nose']   
)
