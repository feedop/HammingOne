#pragma once
#include <list>
#include <chrono>
#include <iostream>

namespace CPU
{
	// Class representing a binary trie
	template<bool printPairs>
	class Trie
	{
	private:
		const unsigned int* input;
		const int sequenceLength;
		const int numberOfSequences;
		inline void search(int index, int integer, int bit, long long& matches);
		inline void insert(int index);

		struct Node
		{
			Node* left = nullptr;
			Node* right = nullptr;
			int rangeCount = 0;
			std::list<int> range;
		} *head;

	public:
		Trie(const unsigned int* input, const int sequenceLength, const int numberOfSequences);
		void hammingOne();
	};

	template <bool printPairs>
	inline void Trie<printPairs>::insert(const int index)
	{
		Node* ptr = head;
		for (int i = 0; i < sequenceLength; i++)
		{
			for (int j = INTSIZE - 1; j >= (i == sequenceLength - 1 ? 1 : 0); j--)
			{
				// right
				if ((input[i * numberOfSequences + index]) & (1 << j))
				{
					if (ptr->right == nullptr)
						ptr->right = new Node();

					ptr = ptr->right;
				}
				else // left
				{
					if (ptr->left == nullptr)
						ptr->left = new Node();

					ptr = ptr->left;
				}
			}
		}
		// last bit
		if ((input[(sequenceLength - 1) * numberOfSequences + index]) & 1)
		{
			if (ptr->right == nullptr)
				ptr->right = new Node();

			if (printPairs) ptr->right->range.push_back(index);
			else ptr->right->rangeCount++;
		}
		else // left
		{
			if (ptr->left == nullptr)
				ptr->left = new Node();

			if (printPairs) ptr->left->range.push_back(index);
			else ptr->left->rangeCount++;
		}
	}

	template <bool printPairs>
	inline void Trie<printPairs>::search(const int index, const int integer, const int bit, long long& matches)
	{
		// look for a our sequence with one bit changed in the trie
		Node* ptr = head;
		// unchanged bits on the left
		for (int i = 0; i < integer; i++)
		{
			for (int j = INTSIZE - 1; j >= 0; j--)
			{
				if ((input[i * numberOfSequences + index]) & (1 << j))
				{
					ptr = ptr->right;
				}
				else
				{
					ptr = ptr->left;
				}

				if (ptr == nullptr) return;
			}
		}
		// same integer, before changed bit
		for (int j = INTSIZE - 1; j > bit; j--)
		{
			if ((input[integer * numberOfSequences + index]) & (1 << j))
			{
				ptr = ptr->right;
			}
			else
			{
				ptr = ptr->left;
			}

			if (ptr == nullptr) return;
		}
		// bit change
		if ((input[integer * numberOfSequences + index]) & (1 << bit))
		{
			ptr = ptr->left;
		}
		else
		{
			ptr = ptr->right;
		}
		if (ptr == nullptr)	return;

		// same integer, after changed bit
		for (int j = bit - 1; j >= 0; j--)
		{
			if ((input[integer * numberOfSequences + index]) & (1 << j))
			{
				ptr = ptr->right;
			}
			else
			{
				ptr = ptr->left;
			}

			if (ptr == nullptr) return;
		}
		//unchanged bits on the right
		for (int i = integer + 1; i < sequenceLength; i++)
		{
			for (int j = INTSIZE - 1; j >= 0; j--)
			{
				if ((input[i * numberOfSequences + index]) & (1 << j))
				{
					ptr = ptr->right;
				}
				else
				{
					ptr = ptr->left;
				}

				if (ptr == nullptr) return;
			}
		}
		// found a match if we arrived at the end
		if (printPairs) matches += ptr->range.size();
		else matches += ptr->rangeCount;

		// print pairs if set to verbose
		if (printPairs)
		{
			char* buf = new char[2 * ptr->range.size() * sequenceLength * INTSIZE + 9];
			int bufCounter = 0;
			for (std::list<int>::iterator it = ptr->range.begin(); it != ptr->range.end(); ++it)
			{
				buf[bufCounter++] = '[';
				buf[bufCounter++] = '\n';
				for (int i = 0; i < sequenceLength; i++)
				{
					for (int j = INTSIZE - 1; j >= 0; j--)
					{
						buf[bufCounter++] = '0' + ((input[i * numberOfSequences + index] >> j) & 0x0001);
					}
				}
				buf[bufCounter++] = ',';
				buf[bufCounter++] = '\n';
				for (int i = 0; i < sequenceLength; i++)
				{
					for (int j = INTSIZE - 1; j >= 0; j--)
					{
						buf[bufCounter++] = '0' + ((input[i * numberOfSequences + *it] >> j) & 0x0001);
					}
				}
				buf[bufCounter++] = '\n';
				buf[bufCounter++] = ']';
				buf[bufCounter++] = ';';
				buf[bufCounter++] = '\n';
				buf[bufCounter++] = '\0';
				printf(buf);
			}
			delete buf;
		}
	}

	// constructor - build the trie
	template <bool printPairs>
	Trie<printPairs>::Trie(const unsigned int* input, const int sequenceLength, const int numberOfSequences) :
		input(input), sequenceLength(sequenceLength), numberOfSequences(numberOfSequences)
	{
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		head = new Node();
		for (int index = 0; index < numberOfSequences; index++)
		{
			insert(index);
		}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Building the tree completed. Time elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000000.0 << " seconds" << std::endl;
	}

	// search for every possible legal pattern in the trie
	template <bool printPairs>
	void Trie<printPairs>::hammingOne()
	{
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		long long matchCount = 0;
		for (int index = 0; index < numberOfSequences; index++)
			for (int integer = 0; integer < sequenceLength; integer++)
			{
				for (int bit = INTSIZE - 1; bit >= 0; bit--)
				{
					search(index, integer, bit, matchCount);
				}
			}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Search completed. Time elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000000.0 << " seconds" << std::endl
			<< " CPU matches: " << matchCount << std::endl;
	}
}