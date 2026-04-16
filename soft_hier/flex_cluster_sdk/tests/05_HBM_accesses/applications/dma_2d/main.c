/*
 * 2D DMA test: strided load from HBM into L1.
 *
 * HBM layout (fp16, row-major):
 *   Matrix A: ROWS rows, COLS_HBM columns  (all columns present in HBM)
 *
 * Transfer goal: load only the first COLS_LOAD columns of each row into L1.
 *   src_stride = COLS_HBM * sizeof(uint16_t)  — jump a full HBM row between reps
 *   dst_stride = COLS_LOAD * sizeof(uint16_t) — pack loaded elements tightly in L1
 *   size       = COLS_LOAD * sizeof(uint16_t) — bytes copied per repetition
 *   reps       = ROWS                          — one rep per HBM row
 *
 * Expected L1 content after DMA:
 *   l1_buf[r][c] == HBM[r][c]  for 0 <= r < ROWS, 0 <= c < COLS_LOAD
 */

#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_printf.h"

#define ROWS       4                /* number of rows to load              */
#define COLS_HBM   64               /* full row width in HBM (fp16 elems)  */
#define COLS_LOAD  16               /* columns to load per row             */
#define ELEM_BYTES sizeof(uint16_t)

/* L1 destination: ROWS × COLS_LOAD fp16 elements, packed tightly */
static uint16_t l1_buf[ROWS * COLS_LOAD] __attribute__((section(".l1"), aligned(64)));

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
        volatile uint16_t *dst = (volatile uint16_t *)l1_buf;

        printf("[2D DMA] Before transfer — L1 buf[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
               dst[0], dst[1], dst[2], dst[3]);

        size_t row_bytes  = COLS_LOAD * ELEM_BYTES;          /* bytes per rep    */
        size_t src_stride = COLS_HBM  * ELEM_BYTES;          /* HBM row pitch    */
        size_t dst_stride = COLS_LOAD * ELEM_BYTES;          /* L1 row pitch     */

        flex_dma_async_2d(
            (uint64_t)(uintptr_t)l1_buf,  /* dst: L1 buffer                  */
            hbm_addr(0),                  /* src: start of HBM matrix A       */
            row_bytes,                    /* size per rep                     */
            dst_stride,                   /* dst stride between reps          */
            src_stride,                   /* src stride between reps          */
            ROWS                          /* number of repetitions            */
        );

        flex_dma_async_wait_all();

        printf("[2D DMA] After  transfer — first element of each loaded row:\n");
        for (int r = 0; r < ROWS; ++r)
        {
            /* first element of row r in the packed L1 buffer */
            printf("  row %d: 0x%04x 0x%04x ... 0x%04x\n",
                   r,
                   dst[r * COLS_LOAD + 0],
                   dst[r * COLS_LOAD + 1],
                   dst[r * COLS_LOAD + COLS_LOAD - 1]);
        }
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
