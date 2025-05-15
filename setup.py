from setuptools import setup, find_packages
import pathlib
import sys

# === Dependency checks ===
missing = []
try:
    import torch
except ImportError:
    missing.append("torch")
try:
    import mpi4py
except ImportError:
    missing.append("mpi4py")

if missing:
    print(f"ERROR: Missing required packages: {', '.join(missing)}")
    print("Please install them manually before proceeding:")
    sys.exit(1)

from torch.utils.cpp_extension import BuildExtension, CppExtension
import mpi4py

# === Optional ninja ===
try:
    import ninja

    USE_NINJA = True
except ImportError:
    USE_NINJA = False
    print(
        "\033[93m[Warning] 'ninja' is not installed. "
        "Falling back to slower build system. "
        "For faster builds, run: pip install ninja\033[0m"
    )

# === Paths ===
srcpath = pathlib.Path(__file__).parent / "csrc"
sources = list(map(str, srcpath.glob("*.cpp"))) + list(map(str, srcpath.glob("*.cu")))
extra_include_paths = [mpi4py.get_include(), str(srcpath)]

# === Long description ===
long_description = ""
readme_path = pathlib.Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


class BuildExtensionWithNinja(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_ninja=True, **kwargs)


# === Setup ===
setup(
    name="pccl",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"pccl": "pccl"},
    ext_modules=[
        CppExtension(
            name="pccl_mpi_extension",
            sources=sources,
            extra_compile_args=["-O3", f"-I{mpi4py.get_include()}"],
            extra_include_paths=extra_include_paths,
        )
    ],
    cmdclass={"build_ext": BuildExtensionWithNinja},
    install_requires=[],  # torch and mpi4py are manually checked above
    author="Siddharth Singh, Abhinav Bhatele",
    description="Performant Collective Communication Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/pccl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
