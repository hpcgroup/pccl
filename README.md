# PCCL: Performant Collective Communication Library

**Authors:** Siddharth Singh, Abhinav Bhatele

PCCL is a high-performance collective communication library with MPI-based C++/CUDA extensions for PyTorch. It provides efficient implementations of collectives such as `all_gather` and `reduce_scatter` to accelerate distributed training workflows.

---

## 🔧 Installation

Before installing PCCL, you must have the following dependencies **pre-installed**:

- [PyTorch](https://pytorch.org/)
- [mpi4py](https://mpi4py.readthedocs.io/)
- An MPI implementation (e.g., OpenMPI, MPICH)
- A C++17-compatible compiler
- CUDA Toolkit (for GPU support)

🚀 Optional: Speed Up Build with ninja
To enable faster C++/CUDA builds, install ninja before building:

```bash
pip install ninja
```

If ninja is not installed, PCCL will fall back to the default (slower) build system.

📦 Installing PCCL

1. 🔨 Building from Source (Recommended for Development)
Clone the repository and run the following inside the root directory:

```bash
git clone https://github.com/hpcgroup/pccl.git
cd pccl
pip install .
```


For development mode (editable installation):

```bash
pip install -e .
```

⚠️ If you forget to install torch or mpi4py beforehand, installation will fail with a helpful error message.

2. 📥 Installing from PyPI
You can also install PCCL directly from PyPI

```bash
pip install pccl
```

Note: This will install the prebuilt package if available. If no prebuilt wheels are published for your platform, pip will attempt to build from source.


✉️ Citing PCCL
If you use PCCL in your research, please cite our work:

```bibtex
@misc{singh2025bigsendoffhighperformance,
  title={The Big Send-off: High Performance Collectives on GPU-based Supercomputers}, 
  author={Siddharth Singh and Mahua Singh and Abhinav Bhatele},
  year={2025},
  eprint={2504.18658},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2504.18658},
}
```
