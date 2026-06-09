/*
 * RedMulE id/count query test.
 *
 * Verifies the mempool RedMulE topology helpers in runtime/mempool/runtime.h:
 *   - mempool_get_redmule_count()  -> NUM_REDMULE_TILES  (== ARCH_NB_REDMULE_TILES)
 *   - mempool_get_redmule_id()     -> RedMulE tile id of the calling core, or -1
 *
 * Reference: cores whose id is a multiple of (NUM_CORES / NUM_REDMULE_TILES) lead a
 * RedMulE tile (id = core_id / stride); all other cores return (uint32_t)-1.
 *
 * Every core in cluster 0 checks its own id against this formula and ORs any mismatch
 * into a shared L1 flag; core 0 of cluster 0 reports the values and PASS/FAIL.
 */

#include "flex_runtime.h"
#include "flex_printf.h"

/* Shared flag in cluster-0 L1. .l1 is NOLOAD, so it is zeroed at runtime below. */
static volatile uint32_t g_fail __attribute__((section(".l1")));

int main(void)
{
    uint32_t cid     = flex_get_cluster_id();
    uint32_t core_id = mempool_get_core_id();
    uint32_t count   = mempool_get_redmule_count();

    /* Reference id for this core (mirrors mempool_get_redmule_id in runtime.h). */
    uint32_t stride      = (NUM_REDMULE_TILES > 0) ? (NUM_CORES / NUM_REDMULE_TILES) : 1;
    uint32_t expected_id = (NUM_REDMULE_TILES == 0)
                               ? (uint32_t)-1
                               : (((core_id % stride) == 0) ? (core_id / stride)
                                                            : (uint32_t)-1);

    flex_barrier_xy_init();

    if (cid == 0 && flex_is_first_core()) {
        g_fail = 0;
    }
    flex_global_barrier_xy();

    /* Each cluster-0 core verifies its own id and the reported count. */
    if (cid == 0) {
        uint32_t actual_id = mempool_get_redmule_id();
        if (actual_id != expected_id || count != ARCH_NB_REDMULE_TILES) {
            g_fail = 1;
        }
    }
    flex_global_barrier_xy();

    if (cid == 0 && flex_is_first_core()) {
        printf("=== RedMulE id/count query ===\n");
        printf("redmule_count      = %u (expected %u)\n", count,
               (uint32_t)ARCH_NB_REDMULE_TILES);
        printf("cores_per_redmule  = %u\n", stride);
        printf("core 0   -> id %u (expected 0)\n", mempool_get_redmule_id());
        printf("%s\n", g_fail ? "FAIL" : "PASS");
    }

    uint32_t fail = (cid == 0) ? g_fail : 0u;
    flex_global_barrier_xy();
    flex_eoc(fail);
    return 0;
}
