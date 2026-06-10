from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_path = Path(__file__).parent / "requirements.txt"
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="rgc-ei-fit-hh",
    version="0.1.0",
    description=(
        "Biophysical Hodgkin-Huxley models of retinal ganglion cells fit from "
        "extracellular multi-electrode array recordings"
    ),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Amrith Lotlikar, Ian Christopher Tanoh, Andrew Lanpouthakoun",
    author_email="lotlikar@stanford.edu",
    url="https://github.com/amrith1/rgc-ei-fit-hh",
    license="MIT",
    packages=find_packages(),
    package_data={"rgc_models": ["mea_512ch_30um_pitch_elec_coords.npy"]},
    install_requires=read_requirements(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
