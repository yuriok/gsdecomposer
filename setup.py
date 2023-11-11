from setuptools import find_packages, setup

setup(
    name="gsdecomposer",
    version="0.0.0.1",
    description="A deep learning-based decomposition framework for the grain size distributions of clastic sediments.",
    platforms="all",
    author="Yuming Liu",
    author_email="liuyuming@ieecas.cn",
    url="https://github.com/yuriok/gsdecomposer",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "psutil",
        "numpy",
        "scipy",
        "scikit-learn",
        "grpcio",
        "protobuf",
        "xlrd",
        "xlwt",
        "openpyxl",
        "pandas",
        "PySide6>=6.2.0",
        "matplotlib>=3.5.0",
        "SciencePlots",
        "qt-material",
        "torch>=2.1.0",
        "torchsummary",
        "tqdm",
        "QGrain>=0.5.4.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English"

    ],
)
