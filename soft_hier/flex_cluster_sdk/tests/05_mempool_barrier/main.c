/*
 * Test 04: Mempool intra-cluster barrier
 * All 16 cores in each cluster participate in mempool_barrier_init() and
 * mempool_barrier(). Each core writes its core_id to a shared L1 array
 * before the barrier, then core 0 verifies all entries after the barrier.
 * Only cluster 0 is tested; other clusters spin.
 * Expected: "PASS" if all 16 entries are written correctly.
 */
#include "flex_runtime.h"

#define TEST_NUM_CORES 16

volatile uint32_t shared_array[TEST_NUM_CORES] __attribute__((section(".l1")));

int main() {
    uint32_t core_id = mempool_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();

    if (cluster_id != 0) {
        while (1) { asm volatile("wfi"); }
    }

    // Initialize barriers (all cores in cluster 0)
    mempool_barrier_init(core_id);

    // Each core writes its ID
    shared_array[core_id] = core_id + 1;

    // Barrier: wait for all cores to finish writing
    mempool_barrier(TEST_NUM_CORES);

    // Core 0 verifies
    if (core_id == 0) {
        flex_print("=== Test 04: Mempool Barrier ===\n");
        uint32_t errors = 0;
        for (uint32_t i = 0; i < TEST_NUM_CORES; i++) {
            if (shared_array[i] != i + 1) {
                flex_print("ERR at core ");
                flex_print_int(i);
                flex_print(": got ");
                flex_print_int(shared_array[i]);
                flex_print("\n");
                errors++;
            }
        }
        if (errors == 0) {
            flex_print("PASS\n");
        } else {
            flex_print("FAIL: ");
            flex_print_int(errors);
            flex_print(" errors\n");
        }
        flex_eoc(0);
    }

    while (1) { asm volatile("wfi"); }
}
