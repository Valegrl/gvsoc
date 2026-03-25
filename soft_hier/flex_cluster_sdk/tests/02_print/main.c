/*
 * Test 02: Printf from all clusters
 * Each cluster's core 0 prints its cluster ID using flex_print/flex_print_int.
 * Cluster 0 core 0 calls flex_eoc after a brief busy-wait to let others print.
 * Expected output: "Hello from cluster X" for each cluster.
 */
#include "flex_runtime.h"

int main() {
    uint32_t core_id = flex_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();

    if (core_id == 0) {
        flex_print("Hello from cluster ");
        flex_print_int(cluster_id);
        flex_print("\n");
    }

    // Let cluster 0, core 0 do EOC after a delay so other clusters can print
    if (core_id == 0 && cluster_id == 0) {
        for (volatile int i = 0; i < 1000; i++);
        flex_eoc(0);
    }

    while (1) {
        asm volatile("wfi");
    }
}
