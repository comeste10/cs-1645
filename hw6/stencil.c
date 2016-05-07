/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Name: Steven Comer 
 * PittID: sfc15
 * OpenACC parallel 3D stencil. Based on the laplace example by NVIDIA. 
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include "timer.h"

#define N 256

double A[N][N][N];
double Anew[N][N][N];

// Main function
int main(void) {
	int stencil(void);
	printf("OpenAcc 3D Stencil (N=%d)\n",N);	
	stencil();
}

// Stencil function
int stencil() {
    const int n = N;
    const int iter_max = 1000;
 
    const double tol = 1.0e-6;
    double error     = 1.0;
    
    memset(A, 0, n * n * n * sizeof(double));
    memset(Anew, 0, n * n * n * sizeof(double));
        
    for (int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++){
        	A[j][i][0]    = 1.0;
   	     	Anew[j][i][0] = 1.0;
    	}
	}
    
	StartTimer();
    int iter = 0;
    
    #pragma acc data copy(A), create(Anew)
    while (error > tol && iter < iter_max) {
        error = 0.0;

	#pragma acc kernels
       	for(int j = 1; j < n-1; j++) {
		for(int i = 1; i < n-1; i++) {
				for(int k = 1; k < n-1; k++){ 
            	   	Anew[j][i][k] = (A[j][i+1][k] + A[j][i-1][k]
                   	                + A[j-1][i][k] + A[j+1][i][k] 
									+ A[j][i][k-1] + A[j][i][k+1]
									+ A[j][i][k]) / 7.0;
					error = fmax( error, fabs(Anew[j][i][k] - A[j][i][k]));  
				}
           	}
       	}
        
	#pragma acc kernels
        for(int j = 1; j < n-1; j++){
            for(int i = 1; i < n-1; i++){
            	for(int k = 1; k < n-1; k++){
                	A[j][i][k] = Anew[j][i][k];   
				}
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000);
	return 0;
}

