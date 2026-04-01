/*
 * Test 07: Data NoC remote TCDM access - incremental debug
 */
#include "flex_runtime.h"

volatile uint32_t test_data __attribute__((section(".l1")));

int main() {
    uint32_t core_id = mempool_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t num_cores = ARCH_NUM_CORE_PER_CLUSTER;

    mempool_barrier_init(core_id);
    if (core_id == 0) {
        test_data = cluster_id + 100;
    }
    mempool_barrier(num_cores);
    flex_barrier_init();

    // All synced
    flex_global_barrier();

    if (core_id == 0 && cluster_id == 0) {
        flex_print("S1 synced\n");

        // Print the remote address we'll try to read
        uint32_t raddr = remote_cid(1, (uint32_t)&test_data);
        flex_print("S2 raddr=");
        flex_print_int(raddr);
        flex_print("\n");

        // Now try the read
        flex_print("S3 reading...\n");
        volatile uint32_t *rptr = (volatile uint32_t *)raddr;
        uint32_t val = *rptr;
        flex_print("S4 val=");
        flex_print_int(val);
        flex_print("\n");
        flex_eoc(0);
    }

    while (1) { asm volatile("wfi"); }
}
