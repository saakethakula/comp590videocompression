# Assignment 1 README

This compressor is lossless, encodes pixels one at a time, and uses exactly 256 arithmetic coding contexts.

## Approach

Pixels are processed in raster order on grayscale frames. For each pixel, I build a predictor from causal neighbors in the current frame and the co-located pixel in the previous frame, left, above, upper left, previous-frame same-position pixel.

The left/above/upper left values are combined with a clamped gradient predictor, then averaged with the previous-frame pixel. The resulting predicted pixel value in [0, 255] selects one of the 256 arithmetic coding contexts.

The encoded symbol is a residual:

residual = current_pixel - prediction (mod 256)

Decoding is exact because the decoder reproduces the same predictor from already decoded pixels and reconstructs:


current_pixel = prediction + residual mod 256


## Compression Results

I ran:

```bash
cargo run -p assgn1 -- -check_decode -count 10 -in data/bourne.mp4 -out data/out.dat
```

Output:

10 frames encoded, average size (bits): 4422692, compression ratio: 3.75


