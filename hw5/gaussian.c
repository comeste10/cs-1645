/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Name: Steven Comer
 * PittID: sfc15
 * OpenMP parallel gaussian elimination. 
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Main routine
int main (int argc, char *argv[]){
	int k, t, i, j, N, T;
	double pivot, ff, **a, time;
	struct timeval timerStart, timerStop, timerElapsed;

	// process command-line parameters
	if(argc < 3){
		printf("ERROR, usage: %s <N> <T>\n", argv[0]);
		exit(-1);
	} else {
		N = atoi(argv[1]);
		T = atoi(argv[2]);
	}
	omp_set_num_threads(T);
	printf("%s (N=%d, T=%d)\n", argv[0], N, T);

	// initialize matrix a
	a = (double **) malloc(sizeof(double *) * N);
	for(i = 0; i < N; i++){
		a[i] = (double *) malloc(sizeof(double) * N);
	}

	// start timer
	gettimeofday(&timerStart, NULL);

	// TO DO: add OpenMP directives
	for(k = 0; k < N; k++) {
    		pivot = a[k][k];
		//#pragma omp parallel for
		{
			for(t = k; t < N; t++)
				a[k][t] = a[k][t] / pivot;
		}
		#pragma omp parallel for private(ff, j)
		{
			for(i = k + 1; i < N; i++) {
				ff = a[i][k];
				//#pragma omp parallel for
				{
					for(j = 0; j < N; j++)
						a[i][j] = a[i][j] - ff * a[k][j];
				}
			}
		}
	}

	// end timer and report execution time
	gettimeofday(&timerStop, NULL);
        timersub(&timerStop, &timerStart, &timerElapsed);
        time = timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
	printf("Elapsed time: %.2f ms\n", time);

	// deallocate matrix a
	for(i = 0; i < N; i++){
		free(a[i]);
	}
	free(a);


}
