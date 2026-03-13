#
# Copyright (C) 2020 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gvsoc.runner
import gvsoc.systree

import interco.router as router
import memory.memory as memory
from pulp.mempool.dma.mempool_dma import MemPoolDma
from elftools.elf.elffile import *
from pulp.mempool.mempool_cluster import Cluster
from pulp.chips.flex_cluster.cluster_registers import ClusterRegisters

import gvsoc.runner
import math
import utils.loader.loader


GAPY_TARGET = True

#Function to get EoC entry
def find_binary_entry(elf_filename):
    # Open the ELF file in binary mode
    with open(elf_filename, 'rb') as f:
        elffile = ELFFile(f)

        # Find the symbol table section in the ELF file
        for section in elffile.iter_sections():
            if isinstance(section, SymbolTableSection):
                # Iterate over symbols in the symbol table
                for symbol in section.iter_symbols():
                    # Check if this symbol's name matches "tohost"
                    if symbol.name == '_start':
                        # Return the symbol's address
                        return symbol['st_value']

    # If the symbol wasn't found, return None
    return None



class Area:

    def __init__(self, base, size):
        self.base = base
        self.size = size



class ClusterArch:
    def __init__(self,  nb_core_per_cluster, 
                        base, 
                        cluster_id,
                        reg_base,            
                        reg_size,
                        num_cluster_x,       
                        num_cluster_y, 
                        auto_fetch=False):

        self.base                   = base
        self.cluster_id             = cluster_id
        self.auto_fetch             = auto_fetch
        self.reg_area               = Area(reg_base, reg_size)

        # Cluster configuration
        self.async_l1_interco           = False
        self.terapool                   = False
        self.nb_cores_per_tile          = 4
        self.nb_sub_groups_per_group    = 1
        self.nb_groups                  = 4
        self.total_cores                = nb_core_per_cluster
        self.bank_factor                = 4
        self.axi_data_width             = 64
        self.nb_axi_masters_per_group   = 1

        #Global Information
        self.num_cluster_x          = num_cluster_x
        self.num_cluster_y          = num_cluster_y

class ClusterUnit(gvsoc.systree.Component):

    def __init__(self, parent, name, arch, binary, parser, entry=0, auto_fetch=True):
        super().__init__(parent, name)

        #Mempool cluster
        mempool_cluster=Cluster( self, 'mempool_cluster', 
                                 async_l1_interco=arch.async_l1_interco, 
                                 terapool=arch.terapool, 
                                 parser=parser, 
                                 nb_cores_per_tile=arch.nb_cores_per_tile,
                                 nb_sub_groups_per_group=arch.nb_sub_groups_per_group, 
                                 nb_groups=arch.nb_groups, 
                                 total_cores=arch.total_cores, 
                                 bank_factor=arch.bank_factor,
                                 axi_data_width=arch.axi_data_width, 
                                 nb_axi_masters_per_group=arch.nb_axi_masters_per_group)

        # Boot Rom
        rom = memory.Memory(self, 'rom', size=0x1000, width_log2=(arch.axi_data_width - 1).bit_length(), stim_file=self.get_file_path('pulp/chips/spatz/rom.bin'))

        # Cluster CSRs
        cluster_regs = ClusterRegisters(self, 'ctrl_registers', wakeup_latency=18 if arch.terapool else 15)

        # DMA
        dma = MemPoolDma(self, 'dma', loc_base=0x0, loc_size=0x400000, tcdm_width=arch.total_cores*arch.bank_factor*4)

        #DMA data
        #To emulate distributed backends in groups
        self.bind(dma, 'axi_read', mempool_cluster, 'dma_axi')
        self.bind(dma, 'axi_write', mempool_cluster, 'dma_axi')
        self.bind(dma, 'tcdm_read', mempool_cluster, 'dma_tcdm')
        self.bind(dma, 'tcdm_write', mempool_cluster, 'dma_tcdm')

        ###################
        ## INTERCONNECTS ##
        ###################

        nb_axi_masters = arch.nb_axi_masters_per_group * arch.nb_groups

        # Wide SoC aggregation router
        wide_soc_router = router.Router(self, 'wide_soc_router', bandwidth=arch.axi_data_width // 8)

        # Cluster Interconnect
        cluster_ico = router.Router(self, 'cluster_ico')

        # DMA, CSRs Interconnect
        periph_ico = router.Router(self, 'periph_ico', bandwidth=4)

        # AXI Interconnect
        axi_ico = []
        for i in range(0, nb_axi_masters):
            axi_ico.append(router.Router(self, f'axi_ico_{i}', latency=0))
            axi_ico[i].o_MAP(wide_soc_router.i_INPUT(), base=0x80000000, size=0x1000000)
            axi_ico[i].o_MAP(cluster_ico.i_INPUT(), rm_base=False)
            self.bind(mempool_cluster, 'axi_%d' % i, axi_ico[i], 'input')

        # Router -> Cluster wide SoC port
        wide_soc_router.o_MAP(self.i_WIDE_SOC())

        # cluster interconnect -> Bootrom
        cluster_ico.o_MAP(rom.i_INPUT(), base=0xa0000000, size=0x10000, latency=1, rm_base=True)

        # cluster interconnect -> periph interconnect
        cluster_ico.o_MAP(periph_ico.i_INPUT(), base=0x40000000, size=0x20000, latency=1, rm_base=False)

        # periph interconnect -> CSR
        periph_ico.o_MAP(cluster_regs.i_INPUT(), base=0x40000000, size=0x10000, latency=1, rm_base=True)

        # periph interconnect -> DMA Ctrl
        periph_ico.add_mapping('dma', base=0x40010000, size=0x10000, latency=1, remove_offset=0x40010000)
        self.bind(periph_ico, 'dma', dma, 'input')

        # narrow_soc
        periph_ico.o_MAP(self.i_NARROW_SOC())

        ## SYNCHRONIZATION

        # Per-cluster binary loader
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)

        # Loader router for directing binary sections to appropriate memories
        loader_router = router.Router(self, 'loader_router', bandwidth=32, latency=1)
        loader.o_OUT(loader_router.i_INPUT())
        loader_router.o_MAP(rom.i_INPUT(), base=0xa0000000, size=0x10000, rm_base=True)
        loader_router.o_MAP(wide_soc_router.i_INPUT())

        # Loader start -> cluster registers (instruction preheat done)
        loader.o_START(cluster_regs.i_INST_PREHEAT_DONE())

        # Loader entry -> mempool cluster (boot address for cores)
        self.bind(loader, 'entry', mempool_cluster, 'loader_entry')

        # HBM preload done -> cluster registers
        self.o_HBM_PRELOAD_DONE(cluster_regs.i_HBM_PRELOAD_DONE())

        # Cluster registers fetch start -> mempool cluster (fetch enable for cores)
        self.bind(cluster_regs, 'fetch_start', mempool_cluster, 'loader_start')

      	#Cluster Registers for synchronization barrier
        for i in range(0, arch.total_cores):
            self.bind(cluster_regs, f'barrier_ack', mempool_cluster, f'barrier_ack_{i}')

    def i_WIDE_SOC(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'wide_soc', signature='io')

    def o_WIDE_SOC(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('wide_soc', itf, signature='io')

    def i_NARROW_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'narrow_input', signature='io')

    def o_NARROW_INPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('narrow_input', itf, signature='io', composite_bind=True)

    def i_NARROW_SOC(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'narrow_soc', signature='io')

    def o_NARROW_SOC(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('narrow_soc', itf, signature='io')

    def i_GLOBAL_SYNC(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'sync_input', signature='io')

    def o_GLOBAL_SYNC(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('sync_input', itf, signature='io', composite_bind=True)

    def i_HBM_PRELOAD_DONE(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'hbm_preload_done', signature='wire<bool>')

    def o_HBM_PRELOAD_DONE(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('hbm_preload_done', itf, signature='wire<bool>', composite_bind=True)