/*
 * Test 05: Barrier config register reads at ARCH_CLUSTER_REG_BASE (0x20000000)
 * Tests: flex_get_cluster_id(), flex_get_enable_value(), flex_get_disable_value(),
 *        flex_get_barrier_num_cluster(), flex_get_barrier_num_cluster_x/y()
 * Each cluster's core 0 reads and prints its own values.
 * Expected for 2x2 config:
 *   cluster 0: id=0, enable=1, disable=0, num=4, nx=2, ny=2
 *   cluster 1: id=1, ...
 *   cluster 2: id=2, ...
 *   cluster 3: id=3, ...
 */
#include "flex_runtime.h"

int main() {
    uint32_t core_id = flex_get_core_id();
    uint32_t cluster_id = flex_get_cluster_id();

    if (core_id == 0) {
        flex_print("=== Test 05: Barrier Config Regs (CID=");
        flex_print_int(cluster_id);
        flex_print(") ===\n");

        flex_print("  cluster_id=");
        flex_print_int(cluster_id);
        flex_print("\n");

        flex_print("  enable_value=");
        flex_print_int(flex_get_enable_value());
        flex_print("\n");

        flex_print("  disable_value=");
        flex_print_int(flex_get_disable_value());
        flex_print("\n");

        flex_print("  num_cluster=");
        flex_print_int(flex_get_barrier_num_cluster());
        flex_print("\n");

        flex_print("  num_cluster_x=");
        flex_print_int(flex_get_barrier_num_cluster_x());
        flex_print("\n");

        flex_print("  num_cluster_y=");
        flex_print_int(flex_get_barrier_num_cluster_y());
        flex_print("\n");

        // Verify expected values
        uint32_t errors = 0;
        if (flex_get_enable_value() != 1) errors++;
        if (flex_get_disable_value() != 0) errors++;
        if (flex_get_barrier_num_cluster() != ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y) errors++;
        if (flex_get_barrier_num_cluster_x() != ARCH_NUM_CLUSTER_X) errors++;
        if (flex_get_barrier_num_cluster_y() != ARCH_NUM_CLUSTER_Y) errors++;

        if (errors == 0) {
            flex_print("  PASS\n");
        } else {
            flex_print("  FAIL: ");
            flex_print_int(errors);
            flex_print(" mismatches\n");
        }
    }

    // Cluster 0 core 0 exits after a delay
    if (core_id == 0 && cluster_id == 0) {
        for (volatile int i = 0; i < 1000; i++);
        flex_eoc(0);
    }

    while (1) { asm volatile("wfi"); }
}
