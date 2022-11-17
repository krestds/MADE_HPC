#include <stdlib.h>
#include <cstring>


/* create 2D matrix */
double *init_matrix(int N){
	double *matrix = new double[N * N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i * N + j] = rand();

	return matrix;
}


/* create vector */
double *init_vector(int N){
	double *vector = new double[N];

	for (int i = 0; i < N; i++)
		vector[i] = rand();

	return vector;
}

/* Matrix multiplication */
double *matrix_multiply(double *mat_1, double *mat_2, int N){
	double *res = new double[N * N];

	std::memset(res, 0, N * N * sizeof(double));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
            			res[j * N + i] += mat_1[j * N + k] * mat_2[k * N  + i];

	return res;
}


/* Vector multiplication  */
double *vector_multiply(double *matrix, double *vector, int N){
	double *res = new double[N];
	
	std::memset(res, 0, N * sizeof(double));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			res[j] += matrix[j * N + i] * vector[i];

    return res;
}
