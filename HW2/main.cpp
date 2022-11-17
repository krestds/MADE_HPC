#include <iostream>
#include <time.h>
#include "matrix.h"

/* using namespace std */

int main(){ 
	clock_t start_t, end_t;
	int sizes[] = {500, 512, 1000, 1024, 2000, 2048};
	int N;

	for (int i = 0; i < 6; i++ )
		N = sizes[i];
		double *matrix = init_matrix(N);
		double *vector = init_vector(N);

		start_t = clock();
		matrix_multiply(matrix, matrix, N);
		vector_multiply(matrix, vector, N);
		end_t = clock();

		double duration = double (end_t - start_t) / CLOCKS_PER_SEC;

		std::cout << "Size = " << N << " Time execution: " << duration << " " << std::ends;

	return 0;

}

