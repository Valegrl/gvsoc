/*
 * Reduction DMA test.
 *
 * Cluster 0's DM core:
 *   1. Fills l1_src with a known uint16 pattern (loaded from HBM).
 *   2. Fills l1_addend with a second known pattern (from HBM offset B).
 *   3. Copies l1_addend → l1_result via 1D DMA (sets the initial accumulator).
 *   4. Issues a REDADD_UINT_16 reduction: l1_src → l1_result.
 *      (self-cluster mask: row=0x1, col=0x1; in multi-cluster sim widen masks)
 *   5. Waits and verifies: l1_result[i] == l1_addend[i] + l1_src[i] (uint16 wrapping).
 *
 * The test exercises:
 *   - FLEX_DMA_MODE_REDUCE programming path in FlexMemPoolDmaCtrl
 *   - DMMASK + DMCPYC (collective_type = 2 = REDADD_UINT_16) forwarding
 */

#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_printf.h"
#include <stdint.h>

#define TRANSFER_ELEMS  32
#define ELEM_BYTES      sizeof(uint16_t)
#define TRANSFER_BYTES  (TRANSFER_ELEMS * ELEM_BYTES)

/* HBM offset for the second data pattern (non-overlapping with matrix A at 0) */
#define HBM_ADDEND_OFFSET  (64 * 64 * sizeof(uint16_t))   /* after matrix A */

static uint16_t l1_src   [TRANSFER_ELEMS] __attribute__((section(".l1"), aligned(64)));
static uint16_t l1_addend[TRANSFER_ELEMS] __attribute__((section(".l1"), aligned(64)));
static uint16_t l1_result[TRANSFER_ELEMS] __attribute__((section(".l1"), aligned(64)));

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
        /* Step 1: load source from HBM matrix A (pattern A) */
        flex_dma_async_1d((uint64_t)(uintptr_t)l1_src, hbm_addr(0), TRANSFER_BYTES);
        flex_dma_async_wait_all();

        /* Step 2: load addend from HBM matrix B (pattern B) */
        flex_dma_async_1d((uint64_t)(uintptr_t)l1_addend,
                          hbm_addr(HBM_ADDEND_OFFSET), TRANSFER_BYTES);
        flex_dma_async_wait_all();

        printf("[Reduction] l1_src[0..3]:    0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_src[0],    l1_src[1],    l1_src[2],    l1_src[3]);
        printf("[Reduction] l1_addend[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_addend[0], l1_addend[1], l1_addend[2], l1_addend[3]);

        /* Step 3: initialise the accumulator (l1_result = l1_addend) */
        flex_dma_async_1d((uint64_t)(uintptr_t)l1_result,
                          (uint64_t)(uintptr_t)l1_addend, TRANSFER_BYTES);
        flex_dma_async_wait_all();

        /* Step 4: reduction-add  l1_src into l1_result
         *   collective_type = COLLECTIVE_REDADD_UINT_16 (enum value 0)
         *   mask: self-cluster only (row=0x1, col=0x1)                    */
        uint16_t row_mask = 0x1;
        uint16_t col_mask = 0x1;

        bare_dma_start_1d_reduction(
            (uint64_t)(uintptr_t)l1_result,   /* dst accumulator              */
            (uint64_t)(uintptr_t)l1_src,      /* src addend                   */
            TRANSFER_BYTES,
            COLLECTIVE_REDADD_UINT_16,
            row_mask, col_mask
        );
        flex_dma_async_wait_all();

        printf("[Reduction] l1_result[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               l1_result[0], l1_result[1], l1_result[2], l1_result[3]);

        /* Step 5: verify l1_result[i] == (uint16_t)(l1_addend[i] + l1_src[i]) */
        uint32_t errors = 0;
        for (int i = 0; i < TRANSFER_ELEMS; ++i)
        {
            uint16_t expected = (uint16_t)(l1_addend[i] + l1_src[i]);
            if (l1_result[i] != expected)
            {
                if (errors < 4)
                    printf("  mismatch[%d]: got 0x%04x, expected 0x%04x\n",
                           i, l1_result[i], expected);
                errors++;
            }
        }
        printf("[Reduction] Verification: %s (%u errors out of %u)\n",
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
