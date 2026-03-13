#ifndef _EXAMPLE_VECTOR_ADD_H_
#define _EXAMPLE_VECTOR_ADD_H_

#include <math.h>
#include "flex_runtime.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

// Multi-cluster parallel vector addition: Z = X + Y
// DM core in each cluster handles DMA.
// Compute cores perform FP16 addition in parallel.
// Inter-cluster synchronization via global barrier.

#define ELEM_SIZE           2       // FP16
#define TOTAL_VECTOR_LEN    16384
#define TOTAL_VECTOR_BYTES  (TOTAL_VECTOR_LEN * ELEM_SIZE)

#define NUM_CLUSTERS        (ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y)
#define NUM_COMPUTE_CORES   (ARCH_NUM_CORE_PER_CLUSTER - 1)

// Per-cluster slice
#define SLICE_LEN           (TOTAL_VECTOR_LEN / NUM_CLUSTERS)
#define SLICE_BYTES         (SLICE_LEN * ELEM_SIZE)

// Per-core chunk within a cluster
#define ELEMS_PER_CORE      (SLICE_LEN / NUM_COMPUTE_CORES)

// L1 buffer layout (per cluster)
#define X_L1_OFFSET         0
#define Y_L1_OFFSET         (X_L1_OFFSET + SLICE_BYTES)
#define Z_L1_OFFSET         (Y_L1_OFFSET + SLICE_BYTES)

// HBM data layout (contiguous vectors)
#define X_HBM_OFFSET        0
#define Y_HBM_OFFSET        (X_HBM_OFFSET + TOTAL_VECTOR_BYTES)
#define Z_HBM_OFFSET        (Y_HBM_OFFSET + TOTAL_VECTOR_BYTES)

void example_vector_add(){

    uint32_t CID  = flex_get_cluster_id();
    uint32_t core = flex_get_core_id();

    // Phase 1: Global barrier — all clusters synchronized before starting
    flex_global_barrier_xy();

    // Phase 2: Each cluster's DM core loads its slice from HBM to L1
    uint32_t hbm_slice_offset = CID * SLICE_BYTES;

    if (flex_is_dm_core()) {
        flex_dma_async_1d(local(X_L1_OFFSET),
                          hbm_addr(X_HBM_OFFSET + hbm_slice_offset), SLICE_BYTES);
        flex_dma_async_1d(local(Y_L1_OFFSET),
                          hbm_addr(Y_HBM_OFFSET + hbm_slice_offset), SLICE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();  // all cores in cluster wait for DMA

    // Phase 3: Compute cores perform parallel vector add Z[i] = X[i] + Y[i]
    if (!flex_is_dm_core()) {
        volatile _Float16 *x = (volatile _Float16 *)(ARCH_CLUSTER_TCDM_BASE + X_L1_OFFSET);
        volatile _Float16 *y = (volatile _Float16 *)(ARCH_CLUSTER_TCDM_BASE + Y_L1_OFFSET);
        volatile _Float16 *z = (volatile _Float16 *)(ARCH_CLUSTER_TCDM_BASE + Z_L1_OFFSET);

        uint32_t start = core * ELEMS_PER_CORE;
        uint32_t end   = start + ELEMS_PER_CORE;
        if (end > SLICE_LEN) end = SLICE_LEN;

        for (uint32_t i = start; i < end; i++) {
            z[i] = x[i] + y[i];
        }
    }
    flex_intra_cluster_sync();  // all cores wait for compute

    // Phase 4: DM core stores Z back to HBM
    if (flex_is_dm_core()) {
        flex_dma_async_1d(hbm_addr(Z_HBM_OFFSET + hbm_slice_offset),
                          local(Z_L1_OFFSET), SLICE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();

    // Phase 5: Global barrier — all clusters done
    flex_global_barrier_xy();
}

#endif