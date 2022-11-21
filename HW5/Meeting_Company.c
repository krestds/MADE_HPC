// acquaintance in the company
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//selectNextPlayer - Choose_Next_Process
//nextPlayerIndex - next_Process_Ind
//players_arr - process_name_arr
//nextPlayerNumber - next_Process_name
//buffer - process
int  Choose_Next_Process(int *process_name_arr, int  total_processes, int * process_cnt)
{	

	(*process_cnt) ++;

	if (*process_cnt == total_processes)
	{
		return -1;
	}

	int range = total_processes - (*process_cnt);
	//srand(time(NULL));
	int next_Process_Ind = *process_cnt + rand() % range;
	int tmp = process_name_arr[*process_cnt];

	//srand(time(NULL));
	process_name_arr[*process_cnt] = process_name_arr[next_Process_Ind];
	process_name_arr[next_Process_Ind] = tmp;

	return process_name_arr[*process_cnt];
}


int main( int argc, char ** argv)
{	

	int psize, prank;
	MPI_Status status;
	int ierr;
	const int  TAG = 1;
	
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
	srand(time(NULL));

	int *process;

   	process = (int*)malloc( (psize + 1) * sizeof(int) );
   	
   	int *process_name_arr = &process[1];
   	int next_Process_name;
 

	if ((prank == 0 )&&(psize > 1))
	{

		printf(" We have %d processes number\n", psize);
		// Initialize process name array. First process  
		for (int i = 0; i < psize; ++i)
		{
			process_name_arr[i] = i;
		}

		process[0] = 0;

	}	

	else
	{

		MPI_Recv(process, psize + 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
	}

	next_Process_name = Choose_Next_Process(process_name_arr, psize, process);
	printf("Send massege from proсess %d to the next process %d \n" , prank, next_Process_name);
	


	if (next_Process_name > 0)
	{

		MPI_Ssend(process, psize + 1, MPI_INT, next_Process_name, TAG,  MPI_COMM_WORLD);
		printf("Process %d finished, start process %d\n", prank, next_Process_name);

	}
	else 
	{
		printf("All processes met");
	}


	

	free(process);
	ierr = MPI_Finalize();

	return 0;
}
