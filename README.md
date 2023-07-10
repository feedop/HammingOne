# HammingOne

Autor: Filip Skrzeczkowski

In a vector (size *n*) of binary sequences of arbitrary (but constant) length (*l*), the program finds all pairs with Hamming distance equal to 1 (two sequences differing at exactly one index). Time complexity *nl<sup>2</sup>* (compared to naïve *n<sup>2</sup>l*) is achieved through the use of radix trees built on CUDA.

Matchings are treated as ordered pairs so their number may be twice as much compared to what you may be expecting if you consider pairs unordered.

## How to build?

The easiest way to build the project is to use the VS2022 .sln file.

## How to use?

./HammingOne [-c] [-v] [input] 

"input" is mandatory. It is the name of the input file containing the binary sequences. "-c" and "-v" are optional.
 * "-c" - launches a CPU version of the algorithm besides the CUDA one. Warning: very slow.
 * "-v" (verbose) - not only displays the number of pairs but also each pair individually. Better use only for small input.
