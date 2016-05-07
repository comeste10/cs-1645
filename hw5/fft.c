/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Name: Steven Comer
 * PittID: sfc15
 * OpenMP parallel fast-fourier transform (fft)
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// SWAP routine
inline void SWAP(double x, double y) {
	double tmp = x;
	x = y;
	y = tmp;
}

// Main routine
int main(int argc, char * argv[]) {

	double *data, wtemp, wpr, wpi, theta, tempr, tempi, time;

	/* new */
	//int my_counter = 0;
	double *wwr;
	double *wwi;

	int nn, mmax, m, j, istep, i, T, n;
	const int isign = -1;
	struct timeval timerStart, timerStop, timerElapsed;

	// process command-line parameters
	if(argc < 3) {
		printf("ERROR, usage: %s <N> <T>\n", argv[0]);
		exit(-1);
	} else {
		n = atoi(argv[1]);
		T = atoi(argv[2]);
	}
	printf("%s (N=%d, T=%d)\n", argv[0], n, T);

	/* new */
	/* alloc space for wwr and wwi */
	//wwr = (double *)malloc(sizeof(double) * n/2);
	//wwi = (double *)malloc(sizeof(double) * n/2);
	int ii = 0;

	// initialize vector data
	data = (double *)malloc(sizeof(double) * (n << 1));

	// start timer
	gettimeofday(&timerStart, NULL);

	// TO DO: add OpenMP directives
	if(n < 2 || n&(n-1)) exit(1);		// n must be a power of 2
	nn = n << 1;
	j = 1;

	/* A */
	//#pragma omp parallel for private(j, m)
	for (i=1; i<nn; i+=2) {
		if(j > i) {
			SWAP(data[j-1],data[i-1]);
			SWAP(data[j],data[i]);
		}
		m=n;
		while(m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}

	mmax = 2;
	while(nn > mmax) {

		istep = mmax << 1;
		theta = isign * (6.28318530717959/mmax);
		wtemp = sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = sin(theta);

		 /* new */
	        /* alloc space for wwr and wwi */
        	wwr = (double *)malloc(sizeof(double) * nn);
        	wwi = (double *)malloc(sizeof(double) * nn);

		/* new */
		// initialize wwr and wwi
		wwr[1] = 1.0;
		wwi[1] = 0.0;

		/* B */
		//#pragma omp parallel for private (wtemp)
		for(ii=3; ii<nn; ii+=2) {
			wwr[ii] = (wtemp = wwr[ii-2])*wpr - wwi[ii-2]*wpi + wwr[ii-2];
			wwi[ii] = wwi[ii-2]*wpr + wtemp*wpi + wwi[ii-2];
		}

		//my_counter = 0;
		
		/* C */
		#pragma omp parallel for private(m, i, j, tempr, tempi)
		for(m=1; m<mmax; m+=2) {
			
			/* D */
			//#pragma omp parallel for private(j, tempr, tempi)
			for(i=m; i<=nn; i+=istep) {
				j = i + mmax;
				//tempr = wwr[my_counter]*data[j-1] - wwi[my_counter]*data[j];
				//tempi = wwr[my_counter]*data[j] + wwi[my_counter]*data[j-1];
				tempr = wwr[i]*data[j-1] - wwi[i]*data[j];
				tempi = wwr[i]*data[j] + wwi[i]*data[j-1];
				data[j-1] = data[i-1] - tempr;
				data[j] = data[i] - tempi;
				data[i-1] += tempr;
				data[i] += tempi;
			}

			//my_counter++;			

		}

		/* new */
	        // deallocate space for wwr and wwi
        	free(wwr);
        	free(wwi);
		
		mmax = istep;

	}

	// end timer and report execution time
	gettimeofday(&timerStop, NULL);
	timersub(&timerStop, &timerStart, &timerElapsed);
	time = timerElapsed.tv_sec*1000.0 + timerElapsed.tv_usec/1000.0;
	printf("Elapsed time: %.2f ms\n", time);

	// deallocate vector data
	free(data);

	/* new */
	// deallocate space for wwr and wwi
	//free(wwr);
	//free(wwi);

}
