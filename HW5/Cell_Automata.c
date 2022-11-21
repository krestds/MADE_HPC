
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>


int CYCLE = 1;

const int rule110[] = {0, 1, 1, 1, 0, 1, 1, 0};
const int rule90[] = {0, 1, 0, 1, 1, 0, 1, 0};

//choose the rule
#define CUR_RULE rule90

int N_CELLS = 100;
int N_STEPS = 50;


void initialization(int* step0, int N, int prank) 
{

	srand(time(0) + prank);
  step0[0] = step0[N - 1] = 0;

  for (size_t i = 1; i < N - 1; i++)
	{
		step0[i] = rand() % 2;
	}
        
}


void rule(int* step0, int* step1, const int* xrule) 
{

	int bcode = *(step0 + 1) + 2 * (*step0) + 4 * (*(step0 - 1));  
  *step1 = xrule[bcode];

}


void exchange(int* step0, int prank, int psize, int pncells) 
{

	//work with cycle
  int pos_left = (prank - 1 + psize) % psize; 
  int pos_right = (prank + 1 + psize) % psize;
	
  MPI_Request rq[4];
  MPI_Irecv(step0, 1, MPI_INT, pos_left, 1, MPI_COMM_WORLD, rq + 0);
	MPI_Isend(step0 + 1, 1, MPI_INT, pos_left, 1, MPI_COMM_WORLD, rq + 1);
	MPI_Isend(step0 + pncells, 1, MPI_INT, pos_right, 1, MPI_COMM_WORLD, rq + 2);
	MPI_Irecv(step0 + pncells + 1, 1, MPI_INT, pos_right, 1, MPI_COMM_WORLD, rq + 3);
	MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
    
  if (!CYCLE) 
	{

  	if (prank == 0)
		{
			step0[0] = 0;
		}
          
    else if (prank == psize - 1)
		{
			step0[pncells + 1] = 0;
		}

  }

}


void cell_communicate(int* uglobal, int* step0, int prank, int psize, int pncells) 
{

	if (prank == 0)
	{
		memcpy(uglobal, step0 + 1, pncells * sizeof(int));
    for (int pfrom = 1; pfrom < psize - 1; ++pfrom)
		{
			MPI_Recv(uglobal + pfrom * pncells, pncells, MPI_INT, pfrom, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
    MPI_Recv(uglobal + (psize - 1) * pncells, pncells + N_CELLS % psize, MPI_INT, psize - 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
  } 
	else
	{
  	MPI_Send(step0 + 1, pncells, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

}


void print_field(int* a, int n) 
{

	for (int i = 0; i < n; ++i)
	{
		if (a[i] == 1)
		{
			printf("#");
		}
		else
		{
			printf("_");
		}
	}

	printf("\n");

}


int main(int argc, char **argv) 
{
	
	int psize, prank;
	int ierr;
	int* step0;
	int* step1;
	
	ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	
	assert(psize > 1);
	
	int pncells = N_CELLS / psize;
	if (prank == psize - 1)
	{
		pncells += N_CELLS % psize;
	}
	
	// ghost cells
	step0 = (int*) malloc((pncells + 2) * sizeof(int));
	step1 = (int*) malloc((pncells + 2) * sizeof(int));

	initialization(step0, pncells + 2, prank);
	step1[0] = 0; 
	step1[pncells + 1] = 0;
	
	int* uglobal;
	if (prank == 0)
	{
		uglobal = (int*) calloc(N_CELLS, sizeof(int));
	}
	
	cell_communicate(uglobal, step0, prank, psize, pncells);
	
	if (prank == 0) 
	{
		print_field(uglobal, N_CELLS);
	}
	
	for (int i = 1; i <= N_STEPS; ++i) 
	{

		exchange(step0, prank, psize, pncells);
	  for (int c = 1; c < pncells + 1; ++c) 
		{
	      rule(&step0[c], &step1[c], CUR_RULE);
	  }
	    
    int* tmp = step0;
    step0 = step1;
    step1 = tmp;

	  cell_communicate(uglobal, step0, prank, psize, pncells);
	    
	  if (prank == 0) 
		{
	  	print_field(uglobal, N_CELLS);
	  }

	}
	

  if (prank == 0)
	{
		free(uglobal);
	}


  free(step0);
  free(step1);
  ierr = MPI_Finalize();

  return 0;

}

