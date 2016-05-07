/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Name: Steven Comer
 * PittID: sfc15
 * OpenACC parallel implicit moving squares method.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include "timer.h"

#define SIGMA 0.008
#define SIGMA_2 SIGMA*SIGMA
#define PHI(dx,dy,dz) exp(-0.5*(pow((dx),2) + pow((dy),2) + pow((dz),2))/SIGMA_2)
#define N 128
#define M 128

// Structure to store properties of input points
typedef struct{
	double x;		// x coordinate
	double y;		// y coordinate
	double z;		// z coordinate
	double nx;		// normal x coordinate
	double ny;		// normal y coordinate
	double nz;		// normal z coordinate
} Point; 

double grid[N][N][N];
Point points[N];

// Main function
int main(void) {
	int implicit(void);
	printf("OpenAcc Implicit Moving Squares (N=%d, M=%d)\n",N,M);	
	implicit();
}

// Implicit moving squares function
int implicit() {
    const int n = N;
	const int m = M;  
	double diff_x, diff_y, diff_z, x, y, z, sum, phi; 
	double dx = 1.0/n;
	double dy = 1.0/n;
	double dz = 1.0/n;
 
    memset(grid, 0, n * n * n * sizeof(double));
        
	StartTimer();

	/*	
	for(int p=0; p<m; p++){
		for(int k=0; k<N; k++){
			for(int j=0; j<N; j++){
				for(int i=0; i<N; i++){
	*/

	/* scatter to gather transformation */
	#pragma acc kernels
	for(int k=0; k<n; k++) {
		for(int j=0; j<n; j++) {
			for(int i=0; i<n; i++) {
				for(int p=0; p<m; p++) {
				
					// generating the coordinates of the grid point
					x = i*dx;
					y = j*dy;
					z = k*dz;

					// computing influence of point p into grid cell i,j,k
					diff_x = x - points[p].x;
					diff_y = y - points[p].y;
					diff_z = z - points[p].z;
					phi = PHI(diff_x,diff_y,diff_z);
					sum = (diff_x * points[p].nx + diff_y * points[p].ny + diff_z * points[p].nz) * phi;

					// updating grid cell
					grid[i][j][k] += sum/phi;
				}
			}
		}
	}

	double runtime = GetTimer();

	printf(" total: %f s\n", runtime / 1000);
	return 0;
}

