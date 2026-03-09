# GVSoC

GVSoC is the PULP chips simulator that is natively included in the Pulp SDK and is described and evaluated fully in Bruschi et al. [\[arXiv:2201.08166v1\]](https://arxiv.org/abs/2201.08166).


## GVSoC documentations

The user documentation, focusing on how to use GVSOC, is available [here](https://gvsoc.readthedocs.io/en/latest/).

The developer documentation, focusing on how to develop models or extend GVSOC is available [here] (https://gvsoc-developer.readthedocs.io/en/latest/).

## GVSoC Tutorial

If you want to learn more about GVSoC, get started through the tutorial available [here](https://gvsoc-developer.readthedocs.io/en/latest/tutorials.html). This tutorial provides hands-on practice on building systems on GVSoC and extracting the performance results.


## OS Requirements installation

These instructions were developed using a fresh Ubuntu 22.04 (Jammy Jellyfish).

The following packages needed to be installed:

~~~~~shell
sudo apt-get install -y build-essential git doxygen python3-pip libsdl2-dev curl cmake gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev wget sphinx-build doxygen
~~~~~

These are the packages neded on a Fedora:

~~~~~shell
sudo dnf install -y make gcc cmake ninja-build.x86_64 g++ pip lz4-devel ccache glibc-devel.i686 zlib-ng-compat-devel.i686 SDL2 SDL2-devel SDL2_ttf-devel.x86_64 SDL2_image-devel.x86_64 wget2 sphinx-build doxygen
~~~~~

Please also check any README.md in the submodules for target-specific requirements, like for example pulp/README.md.

## Python requirements

Additional Python packages are needed and can be installed with the following commands from root folder:

```bash
git submodule update --init --recursive -j8
pip3 install -r core/requirements.txt
pip3 install -r gapy/requirements.txt
```

## Installation

Get submodules and compile GVSoC with this command:

~~~~~shell
make all TARGETS=<target>
~~~~~

<target> should indicate the target for which GVSoC must be build. This can for example be generic targets rv32 or rv64. 

On ETH network, please use this command to get the proper version of gcc and cmake:

~~~~~shell
CXX=g++-14.2.0 CC=gcc-14.2.0 CMAKE=cmake-3.18.1 make all TARGETS=pulp-open
~~~~~

## Usage

The following example can be launched on pulp-open:

~~~~~shell
./install/bin/gvsoc --target=pulp-open --binary examples/pulp-open/hello image flash run
~~~~~

## Citing

If you intend to use or reference GVSoC for an academic publication, please consider citing it:

```
@INPROCEEDINGS{9643828,
	author={Bruschi, Nazareno and Haugou, Germain and Tagliavini, Giuseppe and Conti, Francesco and Benini, Luca and Rossi, Davide},
	booktitle={2021 IEEE 39th International Conference on Computer Design (ICCD)},
	title={GVSoC: A Highly Configurable, Fast and Accurate Full-Platform Simulator for RISC-V based IoT Processors},
	year={2021},
	volume={},
	number={},
	pages={409-416},
	doi={10.1109/ICCD53106.2021.00071}}
```

## Using GVSoC with DRAMsys

If you want to use DRAMsys with GVSoC follow the steps mentioned in [DRAMsys.md](./DRAMSys.md)

---

## Getting Started with SoftHier Simulation 🚀

### Set Up the Environment 🏁

Initialize the simulator environment by running:

```bash
source sourceme.sh
```

### Build and Run the SoftHier Simulation Model 🧱

1. **Build the SoftHier hardware model** 🛠️

   ```bash
   make tmp_hw
   ```

   This performs a clean build, applies the upstream FlooNoC patch, and compiles the `flex_cluster` target.
   The default architecture configuration is `soft_hier/flex_cluster/flex_cluster_arch.py`. To use a custom one:

   ```bash
   cfg=<path/to/your/architecture/configuration/file> make tmp_hw
   ```

2. **Run the simulation** with a binary 🎮

   ```bash
   ./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary <path/to/your/binary.elf> run --trace=/chip/cluster_0/redmule
   ```

   - `--binary`: Specifies the executable binary to load.
   - `--trace`: Selects which component's trace logs are generated.

### Build the Default Binary 💾

To build the default binary from `soft_hier/flex_cluster_sdk/app_example`, run:

```bash
make sw
```

The generated binary `sw_build/softhier.elf` and dump file `sw_build/softhier.dump` are placed in `sw_build/`.

### Build a Custom Binary ✏️

1. Prepare your source code in a folder with a `CMakeLists.txt` that exports sources and include paths:

   ```cmake
   set(SRC_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/main.c
   )

   set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
   set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
   ```

2. Build by pointing `app` to your folder:

   ```bash
   app=<folder/of/your/code> make sw
   ```

### Build Customized Hardware and Software (Highly Recommended) 🧩

The `hs` target builds hardware and software together in one step.
It uses the local FlooNoC sources (without fetching from upstream); run `make tmp_hw` first if you need the upstream FlooNoC patch applied before doing iterative `hs` builds.

```bash
cfg=<path/to/your/architecture/configuration/file> app=<folder/of/your/code> make hs
```

**Example — default app with default architecture:**

```bash
app=soft_hier/flex_cluster_sdk/app_example make hs; make run
```

## SoftHier Visualization 📈

To visualize a SoftHier simulation:

1. Run the simulation with the `runv` target (generates a trace log).
2. Convert the trace to Perfetto format with the `pfto` target.

Integrated example:

```bash
cfg=<arch_cfg> app=<app_folder> make hs runv pfto
```

The trace file is saved at `sw_build/perfetto.json`.  
Open it at 👉 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
