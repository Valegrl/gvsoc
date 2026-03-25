/*
 * Test 01: Basic EOC
 * Simplest test: core 0 of cluster 0 calls flex_eoc(0).
 * All other cores spin in wfi.
 * Expected output: clean exit with return value 0.
 */
#include "flex_runtime.h"

int main() {
    uint32_t core_id = flex_get_core_id();

    if (core_id == 0) {
        flex_eoc(0);
    }

    while (1) {
        asm volatile("wfi");
    }
}
