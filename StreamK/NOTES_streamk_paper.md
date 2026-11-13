# Stream-K paper notes (Sec. 4-5, arXiv:2301.03598)

- "MAC-loop iteration" is the paper's own term (not "work unit" --
  that's ours) for the smallest balanced unit: a CTA-wide
  BLK_M x BLK_N output block updated over one BLK_K slice of k.
- total_iters = tiles_m * tiles_n * iters_per_tile, iters_per_tile = ceil(k/BLK_K)
- iters_per_cta = ceil(total_iters / g) -- every CTA gets this, last one gets the leftover
- linearized m -> n -> k, so a CTA's range can cross tile boundaries
- fixup: CTA that starts a tile (k=0) owns it, writes C. others just
  store partial + signal. owner waits on peers, sums, writes.
- two-tile hybrid (Sec 5.2, what they actually benchmark): ONE FEWER
  full data-parallel wave than floor(t/p) -- that whole held-back
  wave + the leftover tiles get Stream-K-split together, so each
  Stream-K CTA gets >1 tile's worth of iterations but <2.
