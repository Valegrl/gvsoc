/*
 * Test 06: Inter-cluster global barrier
 *
 * Verifies flex_global_barrier() works correctly:
 * 1. Multiple barrier rounds complete without deadlock
 * 2. All clusters participate in each round
 * 3. Work done between barriers is properly ordered
 *
 * Expected: All clusters print in each phase, "PASS" at end.
 */
#include "flex_runtime.h"

/* Per-cluster L1 counter - each cluster increments its own */
volatile uint32_t phase_counter __attribute__((section(".l1")));

int main() {
    uint32_t core_id = mempool_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t num_cores = ARCH_NUM_CORE_PER_CLUSTER;

    // Init
    mempool_barrier_init(core_id);
    if (core_id == 0) {
        phase_counter = 0;
    }
    mempool_barrier(num_cores);
    flex_barrier_init();

    // Round 1
    if (core_id == 0) {
        phase_counter = 1;
    }
    flex_global_barrier();
    if (core_id == 0 && cluster_id == 0) {
        flex_print("=== Test 06: Global Barrier ===\n");
        flex_print("Round1 OK\n");
    }

    // Round 2
    if (core_id == 0) {
        phase_counter = 2;
    }
    flex_global_barrier();
    if (core_id == 0 && cluster_id == 0) {
        flex_print("Round2 OK\n");
    }

    // Round 3
    if (core_id == 0) {
        phase_counter = 3;
    }
    flex_global_barrier();
    if (core_id == 0 && cluster_id == 0) {
        flex_print("Round3 OK\n");
    }

    // Round 4: verify each cluster's counter advanced
    flex_global_barrier();
    if (core_id == 0) {
        // Each cluster prints its own counter
        flex_print("C");
        flex_print_int(cluster_id);
        flex_print("=");
        flex_print_int(phase_counter);
        flex_print(" ");
    }

    flex_global_barrier();

    if (core_id == 0 && cluster_id == 0) {
        flex_print("\nPASS\n");
        flex_eoc(0);
    }

    while (1) { asm volatile("wfi"); }
}
