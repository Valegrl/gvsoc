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
from pulp.mempool.l2_subsystem import L2_subsystem
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
                        insn_base,
                        insn_size, 
                        auto_fetch=False):

        self.base                   = base
        self.cluster_id             = cluster_id
        self.auto_fetch             = auto_fetch
        self.reg_area               = Area(reg_base, reg_size)
        self.insn_area              = Area(insn_base, insn_size)

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

        nb_axi_masters = arch.nb_axi_masters_per_group * arch.nb_groups

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

        # Binary Loader
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary, entry=0x80000000)

        # Originally part of the L3 instruction memory logic in soft_hier/cluster_unit.py
        # L2 Memory
        # Efficient bandwidth of each port is only 1/4 of axi_data_width in current design
        l2_mem = L2_subsystem(self, 'l2_mem', nb_banks=16 if arch.terapool else 4, 
                              bank_width=arch.axi_data_width, size=0x1000000 if arch.terapool else 0x400000, 
                              nb_masters=nb_axi_masters, port_bandwidth=arch.axi_data_width//4)
        
        #Dummy Memory
        dummy_mem = memory.Memory(self, 'dummy_mem', atomics=True, size=0x400000)

        # Instruction router
        instr_router = router.Router(self, 'instr_router', bandwidth=8*arch.total_cores)

        # DMA, CSRs Interconnect
        periph_ico = router.Router(self, 'periph_ico', bandwidth=4)

        # AXI Interconnect
        axi_ico = []
        for i in range(0, nb_axi_masters):
            axi_ico.append(router.Router(self, f'axi_ico_{i}', latency=0))
            axi_ico[i].add_mapping('l2', base=0x80000000, remove_offset=0x80000000, size=0x1000000)
            axi_ico[i].add_mapping('soc')

        soc_ico = router.Router(self, 'soc_ico')    # TODO: output bandwidth only
        soc_ico.add_mapping('bootrom', base=0xa0000000, remove_offset=0xa0000000, size=0x10000, latency=1)
        soc_ico.add_mapping('peripheral', base=0x40000000, size=0x20000, latency=1)
        # Wide SoC path for remote cluster TCDM and HBM accesses via data NoC
        soc_ico.add_mapping('wide_soc', base=0x30000000, size=0x10000000, latency=1)
        soc_ico.add_mapping('wide_soc', base=0xc0000000, size=0x40000000, latency=1)
        # Forward SoC control register window (e.g. EOC) to the narrow SoC path.
        soc_ico.add_mapping('external', base=0x90000000, size=0x10000, latency=1)

        periph_ico = router.Router(self, 'periph_ico', bandwidth=4)
        periph_ico.add_mapping('csr', base=0x40000000, remove_offset=0x40000000, size=0x10000, latency=1)
        periph_ico.add_mapping('dma_ctrl', base=0x40010000, remove_offset=0x40010000, size=0x10000, latency=1)

        # Binary Loader Router
        loader_router = router.Router(self, 'loader_router', bandwidth=32, latency=1)
        loader_router.add_mapping('output')

        ###################
        ## INTERCONNECTS ##
        ###################

        # Group axi port -> axi interconnect
        for i in range(0, nb_axi_masters):
            self.bind(mempool_cluster, 'axi_%d' % i, axi_ico[i], 'input')

        # SoC interconnect
        for i in range(0, nb_axi_masters):
            self.bind(axi_ico[i], 'soc', soc_ico, 'input')

        # Peripheral interconnect
        self.bind(soc_ico, 'peripheral', periph_ico, 'input')

        # FlexCluster virtual router interconnect
        self.bind(soc_ico, 'external', self, 'narrow_soc')
        self.o_NARROW_INPUT(soc_ico.i_INPUT())

        # wide_soc (remote cluster TCDM and HBM via data NoC)
        self.bind(soc_ico, 'wide_soc', self, 'wide_soc')

        # Bootrom
        self.bind(soc_ico, 'bootrom', rom, 'input')

        # L2
        for i in range(0, nb_axi_masters):
            self.bind(axi_ico[i], 'l2', l2_mem, 'input_%d' % i)

        # CSR
        self.bind(periph_ico, 'csr', cluster_regs, 'input')

        # DMA Ctrl
        self.bind(periph_ico, 'dma_ctrl', dma, 'input')

        # Binary loader
        loader.o_OUT(instr_router.i_INPUT())
        loader.o_START(cluster_regs.i_INST_PREHEAT_DONE())
        self.o_HBM_PRELOAD_DONE(cluster_regs.i_HBM_PRELOAD_DONE())

        #loader router
        self.bind(cluster_regs, 'fetch_start', mempool_cluster, 'loader_start')
        self.bind(loader, 'entry', mempool_cluster, 'loader_entry')
        self.bind(loader, 'out', loader_router, 'input')
        loader_router.add_mapping('dummy', base=0x00000000, remove_offset=0x00000000, size=0x400000)
        loader_router.add_mapping('mem', base=0x80000000, remove_offset=0x80000000, size=0x1000000)
        loader_router.add_mapping('rom', base=0xa0000000, remove_offset=0xa0000000, size=0x1000)
        loader_router.add_mapping('csr', base=0x40000000, remove_offset=0x40000000, size=0x10000)
        self.bind(loader_router, 'mem', l2_mem, 'input_loader')
        self.bind(loader_router, 'rom', rom, 'input')
        self.bind(loader_router, 'csr', cluster_regs, 'input')
        self.bind(loader_router, 'dummy', dummy_mem, 'input')

        #Cluster Registers for synchronization barrier
        for i in range(0, arch.total_cores):
            self.bind(cluster_regs, f'barrier_ack', mempool_cluster, f'barrier_ack_{i}')

        #L2 ro-cache configuration
        self.bind(cluster_regs, 'rocache_cfg', mempool_cluster, 'rocache_cfg')

        #DMA data
        #To emulate distributed backends in groups
        self.bind(dma, 'axi_read', mempool_cluster, 'dma_axi')
        self.bind(dma, 'axi_write', mempool_cluster, 'dma_axi')
        self.bind(dma, 'tcdm_read', mempool_cluster, 'dma_tcdm')
        self.bind(dma, 'tcdm_write', mempool_cluster, 'dma_tcdm')

        # Wide input from data_noc -> MemPool TCDM (remote cluster access)
        wide_axi_goto_tcdm = router.Router(self, 'wide_axi_goto_tcdm')
        wide_axi_goto_tcdm.add_mapping('output')
        self.o_WIDE_INPUT(wide_axi_goto_tcdm.i_INPUT())
        self.bind(wide_axi_goto_tcdm, 'output', mempool_cluster, 'dma_tcdm')

    def i_WIDE_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'wide_input', signature='io')

    def o_WIDE_INPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('wide_input', itf, signature='io', composite_bind=True)

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

    def i_SYNC_OUTPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'sync_output', signature='io')

    def o_SYNC_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('sync_output', itf, signature='io')

    def i_SYNC_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'sync_input', signature='io')

    def o_SYNC_INPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('sync_input', itf, signature='io', composite_bind=True)

    def i_HBM_PRELOAD_DONE(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'hbm_preload_done', signature='wire<bool>')

    def o_HBM_PRELOAD_DONE(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('hbm_preload_done', itf, signature='wire<bool>', composite_bind=True)