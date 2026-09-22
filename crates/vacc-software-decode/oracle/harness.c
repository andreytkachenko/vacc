// Regenerates src/rust/oracle_luma.txt (the oracle used by
// `oracle_luma` in src/rust/inter.rs).
//
// The Rust port of decode_inter_luma (edge264_inter.c, C tree deleted in the
// E4 cutover, commit 9cdae1f^; see git history) is bit-exact against this
// machine's GCC/SSE output for all 48 luma modes x 7 (w,h) combos x 3
// weight patterns (336 cases). The oracle file encodes machine-specific
// SIMD behavior (notably the _mm_sra_epi16 floor-division quirk, see
// h264_inter_chroma_contract.md), so regenerate it ON THE MACHINE where the
// Rust tests run.
//
// Replicates exactly the LCG neighborhood generator and buffer geometry of
// the Rust test (stride 32; block top-left = nb[6*32+6]).
//
// Regeneration steps:
//   1. Extract the C tree:
//        git archive 9cdae1f^ crates/vacc-sw-decode/c | tar -x -C /tmp/ziptest3
//      (any directory works; the compile line below wants <dir>/crates/vacc-sw-decode/c/src)
//   2. Build:
//        gcc -O2 -march=native -std=gnu11 -flax-vector-conversions \
//            -I /tmp/ziptest3/crates/vacc-sw-decode/c/src \
//            crates/vacc-software-decode/oracle/harness.c -o /tmp/harness
//   3. Regenerate (target path must be a tracked absolute path):
//        /tmp/harness > <repo>/crates/vacc-software-decode/src/rust/oracle_luma.txt
//
#include <stdint.h>
#include <stdio.h>

#include "edge264_inter.c" // found via -I <extract>/crates/vacc-sw-decode/c/src

static uint32_t rng;
static uint8_t rnd(void) {
	rng = rng * 1664525u + 1013904223u;
	return (uint8_t)(rng >> 24);
}

static const int combos[][2] = {
	{4, 4}, {4, 8}, {8, 4}, {8, 8}, {8, 16}, {16, 8}, {16, 16},
};

// {256,0,0,0,256,256,0,0}, {257,1,1,1,257,257,1,1},
// {pack_w(64,120), 33, 6, 6, pack_w(-30,90), pack_w(20,-50), 45, 99}
static const int16_t wods[3][8] = {
	{256, 0, 0, 0, 256, 256, 0, 0},
	{257, 1, 1, 1, 257, 257, 1, 1},
	{(int16_t)0x7840, 33, 6, 6, (int16_t)0x5AE2, (int16_t)0xCFCE, 45, 99},
};

int main(void) {
	rng = 0x12345678u;
	for (int zi = 0; zi < 3; zi++) {
		for (int ci = 0; ci < 7; ci++) {
			int w = combos[ci][0], h = combos[ci][1];
			int base = (w == 4) ? 0 : (w == 8) ? 16 : 32;
			for (int xy = 0; xy < 16; xy++) {
				static uint8_t nb[64 * 32];
				static uint8_t dstb[64 * 32];
				for (size_t b = 0; b < sizeof nb; b++)
					nb[b] = rnd();
				for (size_t b = 0; b < sizeof dstb; b++)
					dstb[b] = rnd();

				i16x8 wod;
				for (int k = 0; k < 8; k++)
					wod[k] = wods[zi][k];

				// src2/dst = block top-left = nb[6*32+6] / dstb[6*32+6];
				// matches the Rust view (its `src` starts 2*32+2 earlier).
				decode_inter_luma(base + xy, h, 32, nb + 6 * 32 + 6, 32,
											 dstb + 6 * 32 + 6, wod);

				printf("M%d W%d H%d Z%d\n", base + xy, w, h, zi);
				for (int r = 0; r < h; r++) {
					for (int c = 0; c < w; c++)
						printf("%02x", dstb[(6 * 32 + 6 + r * 32) + c]);
					printf("\n");
				}
			}
		}
	}
	return 0;
}
