from setuptools import setup, find_packages

setup(
    name="ioninsertml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "ase>=3.22",
        "scikit-learn>=1.0",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "ioninsertml-evaluate=ioninsertml.bayesian_opt.evaluate:main",
        ],
    },
    python_requires=">=3.9",
)