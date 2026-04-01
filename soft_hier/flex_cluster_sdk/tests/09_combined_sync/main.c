/*
 * Test 08: Combined intra-cluster + inter-cluster synchronization
 *
 * Full integration test:
 * 1. All cores in all clusters init mempool barriers + flex global barrier
 * 2. Phase A: Each core atomically increments a per-cluster L1 counter (intra-cluster work)
 * 3. Intra-cluster barrier
 * 4. Core 0 of each cluster prints its cluster's counter (should equal NUM_CORES)
 * 5. Global barrier
 * 6. Phase B: Cluster 0, core 0 sums all cluster counters via remote TCDM reads
 * 7. Global barrier
 * 8. Verify total = NUM_CLUSTERS * NUM_CORES_PER_CLUSTER
 *
 * Expected: "PASS" if total matches.
 */
#include "flex_runtime.h"

volatile uint32_t cluster_counter __attribute__((section(".l1")));

int main() {
    uint32_t core_id = mempool_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t num_cores = ARCH_NUM_CORE_PER_CLUSTER;
    uint32_t num_clusters = ARCH_NUM_CLUSTER;

    // Init
    mempool_barrier_init(core_id);
    if (core_id == 0) {
        cluster_counter = 0;
    }
    mempool_barrier(num_cores);
    flex_barrier_init();

    // Phase A: each core atomically increments the cluster counter
    __atomic_fetch_add(&cluster_counter, 1, __ATOMIC_RELAXED);

    // Intra-cluster barrier
    mempool_barrier(num_cores);

    // Core 0 prints per-cluster result
    if (core_id == 0) {
        flex_print("C");
        flex_print_int(cluster_id);
        flex_print(" count=");
        flex_print_int(cluster_counter);
        flex_print("\n");
    }

    // Global barrier - all clusters done with Phase A
    flex_global_barrier();

    // Phase B: cluster 0, core 0 sums all counters via data NoC
    if (core_id == 0 && cluster_id == 0) {
        flex_print("=== Test 08: Combined Sync ===\n");

        uint32_t total = cluster_counter; // local cluster 0 value
        for (uint32_t c = 1; c < num_clusters; c++) {
            volatile uint32_t *remote_counter = (volatile uint32_t *)
                remote_cid(c, (uint32_t)&cluster_counter);
            total += *remote_counter;
        }

        flex_print("total=");
        flex_print_int(total);
        flex_print(" expected=");
        flex_print_int(num_clusters * num_cores);
        flex_print("\n");

        if (total == num_clusters * num_cores) {
            flex_print("PASS\n");
        } else {
            flex_print("FAIL\n");
        }
    }

    // Final global barrier before EOC
    flex_global_barrier();

    if (core_id == 0 && cluster_id == 0) {
        flex_eoc(0);
    }

    while (1) { asm volatile("wfi"); }
}
