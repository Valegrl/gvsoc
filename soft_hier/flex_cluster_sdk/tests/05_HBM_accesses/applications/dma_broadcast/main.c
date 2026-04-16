/*
 * Broadcast DMA test.
 *
 * Cluster 0's DM core:
 *   1. Fills l1_src with a known pattern (loaded from HBM).
 *   2. Issues a broadcast DMA: l1_src → l1_dst within the same cluster
 *      (row_mask = 0x1, col_mask = 0x1 → targets cluster (0,0) only).
 *      In a multi-cluster setup widen the masks to fan out to all clusters.
 *   3. Waits for completion and verifies l1_dst matches l1_src.
 *
 * The test exercises:
 *   - FLEX_DMA_MODE_BCAST programming path in FlexMemPoolDmaCtrl
 *   - DMMASK + DMCPYC (collective_type = 1) forwarding to SnitchDma
 */

#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_printf.h"

#define TRANSFER_ELEMS  32          /* fp16 elements per transfer */
#define ELEM_BYTES      sizeof(uint16_t)
#define TRANSFER_BYTES  (TRANSFER_ELEMS * ELEM_BYTES)

/* Two non-overlapping L1 buffers: source and destination */
static uint16_t l1_src[TRANSFER_ELEMS] __attribute__((section(".l1"), aligned(64)));
static uint16_t l1_dst[TRANSFER_ELEMS] __attribute__((section(".l1"), aligned(64)));

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
    {
        /* Step 1: load source data from HBM into l1_src */
        flex_dma_async_1d((uint64_t)(uintptr_t)l1_src, hbm_addr(0), TRANSFER_BYTES);
        flex_dma_async_wait_all();

        printf("[Broadcast] Source l1_src[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_src[0], l1_src[1], l1_src[2], l1_src[3]);
        printf("[Broadcast] Before bcast, l1_dst[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_dst[0], l1_dst[1], l1_dst[2], l1_dst[3]);

        /* Step 2: broadcast l1_src → l1_dst (self-cluster, mask = cluster (0,0)) */
        uint16_t row_mask = 0x1;   /* row 0 only  */
        uint16_t col_mask = 0x1;   /* col 0 only  */

        bare_dma_start_1d_broadcast(
            (uint64_t)(uintptr_t)l1_dst,   /* dst: local L1 destination   */
            (uint64_t)(uintptr_t)l1_src,   /* src: local L1 source        */
            TRANSFER_BYTES,
            row_mask, col_mask
        );
        flex_dma_async_wait_all();

        /* Step 3: verify */
        printf("[Broadcast] After  bcast, l1_dst[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_dst[0], l1_dst[1], l1_dst[2], l1_dst[3]);

        uint32_t errors = 0;
        for (int i = 0; i < TRANSFER_ELEMS; ++i)
        {
            if (l1_dst[i] != l1_src[i]) errors++;
        }
        printf("[Broadcast] Verification: %s (%u errors out of %u)\n",
               errors == 0 ? "PASS" : "FAIL", errors, TRANSFER_ELEMS);
        eoc_val = errors;
    }

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
