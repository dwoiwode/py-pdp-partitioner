from pathlib import Path

import setuptools

from pyPDP import name, version, author_email, description, author

requirements = [
    "configspace>=0.4.20",
    "matplotlib>=3.5.1",
    "numpy>=1.22.0",
    "scipy>=1.7.1",
    "scikit-learn>=1.0",
    "tqdm>=4.62.3",
]

extras_require = {
    "dev": [
        "pytest>=6.2.5",
    ],
    "examples": [
        "pandas>=1.3.5",
        "openml>=0.12.2",
        # "HPOBench @ git+https://github.com/automl/HPOBench.git",
    ]
}
extras_require["test"] = extras_require["dev"] + extras_require["examples"]

setuptools.setup(
    name=name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    version=version,
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"deepcave": ["utils/logging.yml"]},
    python_requires=">=3.8, <3.10",
    install_requires=requirements,
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Windows", "Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
    ],
)
