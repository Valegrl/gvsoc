include test.mk

CMAKE_FLAGS ?= -j 16
CMAKE ?= cmake

TARGETS ?= rv64 \
    rv64_untimed \
    pulp-open \
    pulp-open-nn \
    pulp-open:chip/cluster/redmule=True \
    pulp.spatz.spatz \
    snitch_spatz \
    occamy \
    siracusa \
    snitch \
    ara \
    spatz \
    snitch:core_type=fast \
    pulp.snitch.snitch_cluster_single \
    chimera \
    snitch_testbench \
    magia \
	mempool

ifndef BUILDDIR
ifdef GVSOC_WORKDIR
BUILDDIR = $(GVSOC_WORKDIR)/build
else
BUILDDIR = build
endif
endif

ifndef INSTALLDIR
ifdef GVSOC_WORKDIR
INSTALLDIR = $(GVSOC_WORKDIR)/install
else
INSTALLDIR = install
endif
endif

export PATH:=$(CURDIR)/gapy/bin:$(PATH)

all: checkout build

checkout:
	git submodule update --recursive --init

.PHONY: build mempool_hw config hw

ifdef DEBUG
BUILD_TYPE = RelWithDebInfo
else
BUILD_TYPE = Release
endif

gvrun.build:
	$(CMAKE) -S gvrun -B $(BUILDDIR)/gvrun -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALLDIR)

	cmake --build $(BUILDDIR)/gvrun $(CMAKE_FLAGS)
	cmake --install $(BUILDDIR)/gvrun

build: gvrun.build
	# Change directory to curdir to avoid issue with symbolic links
	cd $(CURDIR) && $(CMAKE) -S . -B $(BUILDDIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALLDIR) \
		-DGVSOC_MODULES="$(CURDIR)/core/models;$(CURDIR)/pulp;$(CURDIR)/gvrun/python;$(MODULES)" \
		-DGVSOC_TARGETS="${TARGETS}" \
		-DCMAKE_SKIP_INSTALL_RPATH=false

	cd $(CURDIR) && $(CMAKE) --build $(BUILDDIR) $(CMAKE_FLAGS)
	cd $(CURDIR) && $(CMAKE) --install $(BUILDDIR)


clean:
	rm -rf $(BUILDDIR) $(INSTALLDIR)

github.test:
	gvtest --testset testset-github.cfg --max-timeout 120 --no-fail run table junit

riscv:
	wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2025.01.17/riscv64-elf-ubuntu-22.04-gcc-nightly-2025.01.17-nightly.tar.xz
	tar xvf riscv64-elf-ubuntu-22.04-gcc-nightly-2025.01.17-nightly.tar.xz

test.withbuild: riscv
	PATH=$(CURDIR)/riscv/bin:$(PATH) && gvtest --testset testset_withbuild.cfg  --thread 1 --no-fail run table junit

doc:
	cd core/docs/user_manual && make html
	cd core/docs/developer_manual && make html
	@echo
	@echo "User documentation: core/docs/user_manual/_build/html/index.html"
	@echo "Developper documentation: core/docs/developer_manual/_build/html/index.html"


######################################################################
## 				Make Targets for DRAMSys Integration 				##
######################################################################

SYSTEMC_VERSION := 2.3.4
SYSTEMC_GIT_URL := https://github.com/accellera-official/systemc.git
SYSTEMC_INSTALL_DIR := $(PWD)/third_party/systemc_install

update:
	cd pulp && git add -N tensorpool.py pulp/mempool/redmule_configurations.py && \
		git diff > ../soft_hier/gvsoc_pulp.patch && \
		git reset -q -- tensorpool.py pulp/mempool/redmule_configurations.py
	cd core && git add -N models/utils/common_cells/or.cpp && \
		git diff -- models/utils/common_cells.py models/utils/common_cells/or.cpp \
		> ../soft_hier/gvsoc_core.patch && \
		git reset -q -- models/utils/common_cells/or.cpp

dramsys_apply_patch:
	git submodule update --init --recursive
	if cd pulp && git apply --check ../soft_hier/gvsoc_pulp.patch; then \
		git apply ../soft_hier/gvsoc_pulp.patch;\
	fi
	if cd core && git apply --check ../soft_hier/gvsoc_core.patch; then \
		git apply ../soft_hier/gvsoc_core.patch;\
	fi
	cp -rf soft_hier/flex_cluster pulp/pulp/chips/flex_cluster


build-systemc: third_party/systemc_install/lib64/libsystemc.so

third_party/systemc_install/lib64/libsystemc.so:
	mkdir -p $(SYSTEMC_INSTALL_DIR)
	cd third_party && \
	git clone $(SYSTEMC_GIT_URL) && \
	cd systemc && git fetch --tags && git checkout $(SYSTEMC_VERSION) && \
	mkdir build && cd build && \
	$(CMAKE) -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$(SYSTEMC_INSTALL_DIR) -DCMAKE_INSTALL_LIBDIR=lib64 .. && \
	make && make install

build-dramsys: build-systemc third_party/DRAMSys/libDRAMSys_Simulator.so

third_party/DRAMSys/libDRAMSys_Simulator.so:
	mkdir -p third_party/DRAMSys
	cp soft_hier/libDRAMSys_Simulator.so third_party/DRAMSys/
	echo "Check Library Functionality"
	cd soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/ && \
	gcc main.c -ldl
	@if soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out ; then \
        echo "Test libaray succeeded"; \
		rm soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out; \
    else \
		rm soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out; \
		rm third_party/DRAMSys/libDRAMSys_Simulator.so; \
		echo "Test libaray failed, We need to rebuild the library, tasks around 40 min"; \
		echo -n "Do you want to proceed? (y/n) "; \
		read -t 30 -r user_input; \
		if [ "$$user_input" = "n" ]; then echo "oops, I see, your time is precious, see you next time"; exit 1; fi; \
		echo "Go Go Go!" ; \
		cd soft_hier/build_dynlib_from_github_dramsys5 && make all; \
		cp DRAMSys/build/lib/libDRAMSys_Simulator.so ../../third_party/DRAMSys/ ; \
		make clean; \
    fi

build-configs: core/models/memory/dramsys_configs

core/models/memory/dramsys_configs:
	cp -rf soft_hier/dramsys_configs core/models/memory/

build-toolchain: third_party/toolchain

third_party/toolchain:
	mkdir -p third_party/toolchain
	cd third_party/toolchain && \
	wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 &&\
	tar -xvjf v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 &&\
	wget https://github.com/husterZC/gun_toolchain/releases/download/v2.0.0/toolchain.tar.xz &&\
	tar -xvf toolchain.tar.xz

third_party/gnu_toolchain:
	mkdir -p third_party/gnu_toolchain
	cd third_party/gnu_toolchain; \
	git clone https://github.com/riscv/riscv-gnu-toolchain; \
	cd riscv-gnu-toolchain/; git reset --hard 935b263; \
	mkdir install; \
	./configure --prefix=$(abspath third_party/gnu_toolchain/riscv-gnu-toolchain/install) --with-arch=rv32gv_zfh --with-abi=ilp32d --with-cmodel=medlow --enable-multilib; \
	make

softhier_preparation: dramsys_apply_patch build-systemc build-dramsys build-configs build-toolchain

clean_preparation:
	rm -rf third_party

######################################################################
## 				Make Targets for SoftHier Simulator 				##
######################################################################

config_file ?= "soft_hier/flex_cluster/flex_cluster_arch.py"
ifdef cfg
	config_file = "$(cfg)"
endif

config:
	rm -rf pulp/pulp/chips/flex_cluster
	cp -rf soft_hier/flex_cluster pulp/pulp/chips/flex_cluster
	cp $(config_file) pulp/pulp/chips/flex_cluster/flex_cluster_arch.py

hw:
	make config
	make TARGETS=pulp.chips.flex_cluster.flex_cluster all

# bowwang: Deeploy-related
hw-deeploy:
	config_file=examples/SoftHier/config/arch_deeploy.py make config
	make TARGETS=pulp.chips.flex_cluster.flex_cluster all-deeploy

######################################################################
## 				Make Targets for Run Simulator		 				##
######################################################################

preload_arg ?= ""
ifdef pld
	pld_path = $(abspath $(pld))
	preload_arg = --preload $(pld_path)
endif

bin ?=
trace_dir ?= traces

.PHONY: check-bin
check-bin:
	@test -n "$(bin)" || { echo "ERROR: set bin=<path/to/app.elf>, e.g. 'bin=path/to/app.elf make run'"; exit 1; }
	@mkdir -p $(trace_dir)

run: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(bin) run $(preload_arg) \
	--trace-level=debug \
	--trace=/chip/cluster_0/dma \
	--trace-level=trace \
	--trace=/chip/cluster_0/dma \
	--trace=/chip/data_noc/ni_1_1 \
	--trace=/chip/data_noc/router_1_1 \
	--trace=/chip/cluster_0/wide_axi_goto_tcdm \
	--trace=/chip/cluster_0/mempool_cluster/dma_tcdm_itf \
	--trace=/chip/cluster_0/mempool_cluster/dma_tcdm_interleaver \
	--trace=/chip/cluster_0/mempool_cluster/dma_axi_interleaver \
	--trace=/chip/cluster_0/mempool_cluster/group_3/tile_0/addr_scrambler3 \
	| tee $(trace_dir)/run_trace.txt

run_only: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(bin) run $(preload_arg) | tee $(trace_dir)/run_trace.txt

runv: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(bin) run $(preload_arg) --trace=redmule --trace=idma --trace=spatz --trace=cluster_registers | tee $(trace_dir)/analyze_trace.txt

runv_simple: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(bin) run $(preload_arg) --trace=redmule --trace=idma --trace=cluster_registers | tee $(trace_dir)/analyze_trace.txt

rund: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(bin) run $(preload_arg) --trace-level=6 --trace=/chip/cluster_4/pe0/insn

run_spatz: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster \
		--binary $(bin) run $(preload_arg) \
		--trace=/chip/cluster_0/pe0/insn \
		--trace=spatz \
		| tee $(trace_dir)/analyze_trace.txt; \
	if [ -n "$(trace_name)" ]; then \
		cp $(trace_dir)/analyze_trace.txt ./$(trace_name)_trace.txt; \
		echo "Copied trace to ./$(trace_name)_trace.txt"; \
	else \
		echo "No trace_name specified. File kept as $(trace_dir)/analyze_trace.txt"; \
	fi

######################################################################
## 				Make Targets for Trace Analyzer		 				##
######################################################################

trace_file ?= $(trace_dir)/analyze_trace.txt
ifdef trace
	trace_file = $(trace)
endif
pfto:
	@mkdir -p $(trace_dir)
	python soft_hier/flex_cluster_utilities/trace_perfetto/parse.py $(trace_file) $(trace_dir)/roi.json
	python soft_hier/flex_cluster_utilities/trace_perfetto/visualize.py $(trace_dir)/roi.json -o $(trace_dir)/perfetto.json


pfto_spatz: check-bin
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster \
		--binary $(bin) run $(preload_arg) \
		--trace-level=trace \
		--trace="/chip/cluster_0/*" \
		| tee $(trace_file); \
	if [ -n "$(trace_name)" ]; then \
		out_name=$(trace_name)_trace; \
	else \
		out_name=perfetto; \
	fi; \
	python soft_hier/flex_cluster_utilities/trace_perfetto/parse.py $(trace_file) $(trace_dir)/roi.json; \
	python soft_hier/flex_cluster_utilities/trace_perfetto/visualize.py $(trace_dir)/roi.json -o ./$$out_name.json; \
	echo "Generated ./$$out_name.json"

clean_trace:
	rm *_trace.json

######################################################################
## 				Make Targets for C2C Platform 						##
######################################################################

ccfg_file ?= $(realpath soft_hier/c2c_platform/c2c_platform_cfg.py)
ifdef ccfg
	ccfg_file = $(realpath $(ccfg))
endif

c2c-cfg:
	rm -rf pulp/pulp/c2c_platform
	cp -rf soft_hier/c2c_platform pulp/pulp/c2c_platform
	cp ${ccfg_file} pulp/pulp/c2c_platform/c2c_platform_cfg.py

c2c-hw: c2c-cfg
	make TARGETS=pulp.c2c_platform.c2c_platform all

c2c-run:
	./install/bin/gvsoc --target=pulp.c2c_platform.c2c_platform run --trace=ctrl --trace=endpoint

######################################################################
## 				Snitch cluster testsuite			 				##
######################################################################

snitch_cluster.checkout:
	git clone git@github.com:pulp-platform/snitch_cluster.git -b gvsoc-ci
	cd snitch_cluster && git submodule update --recursive --init

snitch_cluster.build:
	cd snitch_cluster/target/snitch_cluster && make DEBUG=ON OPENOCD_SEMIHOSTING=ON sw

snitch_cluster.test:
	cd snitch_cluster/target/snitch_cluster && GVSOC_TARGET=$(TARGETS) ./util/run.py sw/run.yaml --simulator gvsoc -j

snitch_cluster: snitch_cluster.checkout snitch_cluster.build snitch_cluster.test