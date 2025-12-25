#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "cuda.h"

using namespace std;

#define	cudaCheckError(error)																	\
if (error != cudaSuccess) {																		\
	cout << cudaGetErrorString(error) << endl;													\
	cout << "In file " << __FILE__ << " line " << __LINE__ << endl;								\
	exit(EXIT_FAILURE);																			\
}																								\
else {																							\
	auto last = cudaGetLastError();																\
	if (last != cudaSuccess) {																	\
		cout << cudaGetErrorString(last) << endl;												\
		cout << "In file " << __FILE__ << " line " << __LINE__ << endl;							\
		exit(EXIT_FAILURE);																		\
	}																							\
}
