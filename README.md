# SoftHier (x MemPool) Simulation Model in GVSoC 🚀

## Attention

**This is the tmp branch only for dev purpose.*
*

## SoftHier Architecture Overview 🏗️

![SoftHier Architecture Diagram](docs/figures/SoftHier_Arch.png)

## OS Requirements Installation 🖥️

The following instructions are designed for a fresh installation of Ubuntu 22.04 (Jammy Jellyfish).

To install the required packages, run:

```bash
sudo apt-get install -y build-essential git doxygen python3-pip libsdl2-dev curl cmake gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
```

## Toolchain and Shell Requirements 🔧

GVSoC requires the following tools and versions:

- **g++** and **gcc** versions >= 11.2.0
- **cmake** version >= 3.18.1
- **Python** version >= 3.11.3

Please ensure your toolchain meets these requirements.

Also please make sure you are using the bash shell for SoftHier Simulation:

```bash
bash
```

## Getting Started with SoftHier Simulation 🚀

> [!WARNING]
> **This repository builds the hardware/simulator only and runs pre-built binaries.**
> It does **not** build application software. Build your application binaries in the external
> MemPool repository — **https://github.com/Valegrl/mempool** (branch `main`) — and then run
> them here with `bin=<path/to/your.elf> make run`.

### Clone the Repository and Set Up the Environment 🏁

Follow these steps to set up the SoftHier simulation environment:

1. **Clone the repository** and navigate into the project directory:

   ```bash
   git clone https://github.com/Valegrl/gvsoc.git -b soft_hier_mempool soft_hier_mempool
   cd soft_hier_mempool
   ```
2. **Initialize the simulator environment** by running:

   ```bash
   source sourceme.sh
   ```

### Build the SoftHier Hardware Model 🧱🛠️

**Build the SoftHier hardware model**:

```bash
   make hw
```

The default configuration file is located at `soft_hier/flex_cluster/flex_cluster_arch.py` and builds the **Mempool** preset (256 cores/cluster). To use a custom architecture configuration, specify the file path as follows:

```bash
   cfg=<path/to/your/architecture/configuration/file> make hw
```

   Four ready-made Mempool-family presets ship in `soft_hier/flex_cluster/configs/`:

```bash
   # Minpool (16 cores/cluster)
   cfg=soft_hier/flex_cluster/configs/arch_minpool.py  make hw

   # Mempool (256 cores/cluster) -- same as the default
   cfg=soft_hier/flex_cluster/configs/arch_mempool.py  make hw

   # Terapool (1024 cores/cluster)
   cfg=soft_hier/flex_cluster/configs/arch_terapool.py make hw

   # Tensorpool (256 cores/cluster + 16 RedMulE tensor engines, sub-groups, 4 MB L1)
   cfg=soft_hier/flex_cluster/configs/arch_tensorpool.py make hw
```

> [!IMPORTANT]
> The architecture you build here (cluster count, cores, L1 sizes, ...) must match the
> configuration the binary was compiled against in the external MemPool repo. A mismatch
> (e.g. a 4×4 binary on a 2×2 model) shows up at runtime as out-of-bound accesses to the
> `debug_mem` sink. Rebuild the binary in the MemPool repo whenever you change the `cfg=` preset here.

## Run a Binary on the Simulator 🎮

Point `bin=` at an ELF built in the external MemPool repository and run it on the SoftHier
model you built above:

```bash
bin=<path/to/your.elf> make run
```

`make run` executes the simulator with the detailed debug traces and writes the log to
`traces/run_trace.txt`. For a clean run without the extra traces, use `run_only`:

```bash
bin=<path/to/your.elf> make run_only
```

You can also invoke the simulator directly:

```bash
./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster \
    --binary <path/to/your.elf> run --trace=/chip/cluster_0 --preload=my_preload.elf
```

- `--binary` (`bin=`): the pre-built ELF to load and execute.
- `--trace`: which component's trace logs to emit during the simulation.
- `--preload` (`pld=`): an additional ELF loaded into memory before execution (e.g. pre-initialized HBM/DRAM data):

```bash
pld=my_preload.elf bin=<path/to/your.elf> make run
```

**Generating HBM preload data.** To build an HBM preload binary from NumPy data, use the
preload helper shipped in this repo (or generate it in the external MemPool repo):

```bash
python soft_hier/flex_cluster_utilities/preload.py
```

## SoftHier Visualization 📈

To visualize a SoftHier simulation, follow these steps:

1. Run with the `runv` Makefile target to produce an analysis trace.
2. Convert it to a Perfetto-format trace with the `pfto` target.

```bash
bin=<path/to/your.elf> make runv pfto
```

The trace file will be saved at:
📂 `traces/perfetto.json`

To view the trace, open the following URL in your browser:
👉 [Perfetto UI](https://ui.perfetto.dev/)
