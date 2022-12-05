#pragma once

namespace GPU
{
	void buildTrie(const unsigned int* input, int* L, int* R, int* outLMinRange, int* outRMinRange, const int numberOfSequences, const int sequenceLength, bool printPairs);
}
