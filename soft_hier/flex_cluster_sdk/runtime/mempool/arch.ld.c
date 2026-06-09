// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Adapted from mempool arch.ld.c for flex_cluster integration.
// This file is preprocessed by GCC (-P -E) to expand macros into arch.ld.

MEMORY {
  l1 (R) : ORIGIN = 0x00000000, LENGTH = (NUM_BANKS * L1_BANK_SIZE)
  l2     : ORIGIN = L2_BASE   , LENGTH = L2_SIZE
  rom (R) : ORIGIN = BOOT_ADDR , LENGTH = 0x00001000
  hbm    : ORIGIN = 0xc0000000, LENGTH = 0x20000000
}

SECTIONS {
  // Start and end of memories
  __l1_start = ORIGIN(l1);
  __l1_end = ORIGIN(l1) + LENGTH(l1);
  __l2_start = ORIGIN(l2);
  __l2_end = ORIGIN(l2) + LENGTH(l2);
  __rom_start = ORIGIN(rom);
  __rom_end = ORIGIN(rom) + LENGTH(rom);

  // Stack region in sequential L1 memory
  __stack_start = __l1_start;
  __stack_end = __l1_start + (NUM_BANKS * STACK_SIZE / BANKING_FACTOR);

  // Sequential region size (stacks + per-tile data)
  __seq_start = __l1_start;
  __seq_end = __l1_start + (NUM_BANKS * STACK_SIZE / BANKING_FACTOR);

  // Heap size (start address is re-assigned in flex_memory.ld)
  __heap_start = __l1_start;
  __heap_end = __l1_end;
}
