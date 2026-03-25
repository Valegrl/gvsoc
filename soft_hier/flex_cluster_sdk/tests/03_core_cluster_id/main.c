/*
 * Test 03: Core and Cluster ID queries
 * Tests flex_get_core_id(), flex_get_cluster_id(), flex_is_first_core(),
 * flex_is_dm_core(), mempool_get_core_id(), mempool_get_tile_id(),
 * mempool_get_group_id(), get_pos().
 * Only cluster 0, core 0 prints all results.
 * Expected: core_id=0, cluster_id=0, is_first=1, is_dm=0,
 *           tile_id=0, group_id=0, pos=(0,0)
 */
#include "flex_runtime.h"

int main() {
    uint32_t core_id = flex_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();

    if (core_id == 0 && cluster_id == 0) {
        flex_print("=== Test 03: ID Queries ===\n");

        flex_print("core_id=");
        flex_print_int(core_id);
        flex_print("\n");

        flex_print("cluster_id=");
        flex_print_int(cluster_id);
        flex_print("\n");

        flex_print("is_first_core=");
        flex_print_int(flex_is_first_core());
        flex_print("\n");

        flex_print("is_dm_core=");
        flex_print_int(flex_is_dm_core());
        flex_print("\n");

        // Mempool-level queries
        flex_print("mempool_core_id=");
        flex_print_int(mempool_get_core_id());
        flex_print("\n");

        flex_print("mempool_tile_id=");
        flex_print_int(mempool_get_tile_id());
        flex_print("\n");

        flex_print("mempool_group_id=");
        flex_print_int(mempool_get_group_id());
        flex_print("\n");

        flex_print("mempool_core_count=");
        flex_print_int(mempool_get_core_count());
        flex_print("\n");

        // FlexCluster position
        FlexPosition pos = get_pos(cluster_id);
        flex_print("pos.x=");
        flex_print_int(pos.x);
        flex_print(" pos.y=");
        flex_print_int(pos.y);
        flex_print("\n");

        flex_print("PASS\n");
        flex_eoc(0);
    }

    while (1) {
        asm volatile("wfi");
    }
}
