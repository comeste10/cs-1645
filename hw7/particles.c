/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Name: Steven Comer
 * PittID: sfc15
 * MPI particle-interaction code. 
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"	/* new, from hw6 */

#define TAG 7
#define CONSTANT 777

// Particle interaction constants
#define A 10250000.0
#define B 726515000.5
#define MASS 0.1
#define DELTA 1

// Random initialization constants
#define POSITION 0
#define VELOCITY 1

// Structure for private properties of a particle
struct privateParticle{
	float initX;
	float initY;
	float initVX;
	float initVY;
	float aX;
	float aY;
	float vX;
	float vY;	
};

// Structure for shared properties of a particle (this data is included in messages)
struct sharedParticle{
	float x;
	float y;
	float mass;
	float fx;
	float fy;
};

// Heading for auxiliar functions
float randomValue(int type);
void printParticles(int rank, struct privateParticle *priv, struct sharedParticle *shared, int limit);
void interact(struct sharedParticle *source, struct sharedParticle *destination);
void evolve(struct sharedParticle *source, struct sharedParticle *destination, int limit);
void merge(struct sharedParticle *first, struct sharedParticle *second, int limit);
void updateProperties(struct privateParticle *priv, struct sharedParticle *shared, int limit);

// Main function
main(int argc, char** argv){
	int myRank;									// Rank of process
	int p;										// Number of processes
	int n;										// Number of particles
	int previous;								// Previous process in the ring
	int next;									// Next process in the ring
	int tag = TAG;								// Tag for message
	int number;									// Number of particles
	struct privateParticle *privateLocals;		// Array of private properties for local particles
	struct sharedParticle *locals;				// Array of local particles
	struct sharedParticle *foreigners;			// Array of foreign particles
	MPI_Status status;							// Return status for receive
	int j,rounds, initiator, sender;

	/* new */
	/* MPI_Request for MPI_Isend */
	MPI_Request request;

	// checking the number of parameters
	if(argc < 2){
		printf("ERROR: Not enough parameters\n");
		printf("Usage: <program> <number of particles>\n");
		exit(1);
	}
	
	// getting number of particles
	n = atoi(argv[1]);

	// initializing MPI structures and checking p is odd
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	if(p % 2 == 0){
		p = p - 1;
		if(myRank == p){
			MPI_Finalize();
			return 0;
		}
	}
	srand(myRank+myRank*CONSTANT);

	// TO DO: compute previous and next ranks
	previous = (myRank - 1 + p) % p;
	next = (myRank + 1) % p;

	// allocating memory for particle arrays
	number = n / p;
	privateLocals = (struct privateParticle *) malloc(number * sizeof(struct privateParticle));
	locals = (struct sharedParticle *) malloc(number * sizeof(struct sharedParticle));
	foreigners = (struct sharedParticle *) malloc(number * sizeof(struct sharedParticle));

	printf("%d\n",number);

	// random initialization of local particle array
	for(j = 0; j < number; j++){
		locals[j].x = randomValue(POSITION);
		locals[j].y = randomValue(POSITION);
		locals[j].fx = 0.0;
		locals[j].fy = 0.0;
		locals[j].mass = MASS;	
		privateLocals[j].vX = randomValue(VELOCITY);
		privateLocals[j].vY = randomValue(VELOCITY);
		privateLocals[j].initVX = privateLocals[j].vX;
		privateLocals[j].initVY = privateLocals[j].vY;
		privateLocals[j].initX = locals[j].x;
		privateLocals[j].initY = locals[j].y;
	}

	/* new */
	
	/* sharedParticle datatype setup */
	MPI_Datatype mpi_sharedParticle;
	MPI_Type_contiguous(5, MPI_FLOAT, &mpi_sharedParticle);
	MPI_Type_commit(&mpi_sharedParticle);

	/* privateParticle datatype setup */
	MPI_Datatype mpi_privateParticle;
	MPI_Type_contiguous(8, MPI_FLOAT, &mpi_privateParticle);
	MPI_Type_commit(&mpi_privateParticle);

/* start timer */
	StartTimer();

	// TO DO: send the local particles to the next processor
	MPI_Isend(locals, number, mpi_sharedParticle, next, tag, MPI_COMM_WORLD, &request);
	// TO DO: receive the incoming foreign particle set into foreigners variable
	MPI_Recv(foreigners, number, mpi_sharedParticle, previous, tag, MPI_COMM_WORLD, &status);
	evolve(locals,foreigners,number);

	// running the algorithm for (p-1)/2 rounds
	for(rounds = 1; rounds < (p-1)/2; rounds++){
		// TO DO: send the foreign particles to the next processor
		MPI_Isend(locals, number, mpi_sharedParticle, next, tag, MPI_COMM_WORLD, &request);
		// TO DO: receive the foreign particles from previous processor
		MPI_Recv(foreigners, number, mpi_sharedParticle, previous, tag, MPI_COMM_WORLD, &status);
		evolve(locals,foreigners,number);
	}

	// TO DO: send the particles to the initiator
	initiator = (myRank - (p/2) + p) % p; 
	MPI_Isend(foreigners, number, mpi_sharedParticle, initiator, tag, MPI_COMM_WORLD, &request);

	// TO DO: receive the incoming particles and merge them with the local set, interacting the local set
	sender = (myRank + (p/2)) % p;
	MPI_Recv(foreigners, number, mpi_sharedParticle, sender, tag, MPI_COMM_WORLD, &status);
	merge(locals,foreigners,number);
	evolve(locals,locals,number);

	// computing new velocity, acceleration and position
	updateProperties(privateLocals,locals,number);

/* end timer */
	double runtime = GetTimer();
	printf("N = %d\tcores = %d\truntime = %f s\n", n, p, runtime/1000);
	MPI_Barrier(MPI_COMM_WORLD);


	// TO DO: change printParticles function to avoid printing data on the screen and instead create a file output_P.txt with all the particle information
	printParticles(myRank,privateLocals,locals,number);

	// disposing memory for particle arrays
	free(locals);
	free(foreigners);

	/* new */	
	/* free dataypes */
	MPI_Type_free(&mpi_sharedParticle);
	MPI_Type_free(&mpi_privateParticle);

	// finalizing MPI structures
	MPI_Finalize();
}

// Function for random value generation
float randomValue(int type){
	float value;
	switch(type){
		case POSITION:
			value = (float)rand() / (float)RAND_MAX * 100.0;
			break;
		case VELOCITY:
			value = (float)rand() / (float)RAND_MAX * 10.0;
			break;
		default:
			value = 1.1;
	}
	return value;
}

// Function for printing out the particle array
void printParticles(int rank, struct privateParticle *priv, struct sharedParticle *shared, int limit){
	
	/* new */
	/* output to a file output_P.txt instead of stdout */
	FILE * outfile;	
	char * filename;
	filename = (char *)malloc(30 * sizeof(char));
	sprintf(filename, "output_%d.txt", rank);
	outfile = fopen(filename, "w");	
	
	/* only change below is printf --> fprintf */
	int j;
	fprintf(outfile,"*******************************************************************\n");
	fprintf(outfile,"Processor %d\n",rank);
	fprintf(outfile,"Part     x0         y0         vx0        vy0        fx         fy         x          y          vx         vy\n");
	for(j = 0; j < limit; j++){
		fprintf(outfile,"%-5d %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E %-10.2E\n",j,priv[j].initX,priv[j].initY,priv[j].initVX,priv[j].initVY,shared[j].fx,shared[j].fy,shared[j].x,shared[j].y,priv[j].vX,priv[j].vY);
	}
	fprintf(outfile,"*******************************************************************\n");

	/* new */
	/* cleanup */
	fclose(outfile);
	free(filename);
}

// Function for computing interaction among two particles
// There is an extra test for interaction of identical particles, in which case there is no effect over the destination
void interact(struct sharedParticle *first, struct sharedParticle *second){
	float rx,ry,r,fx,fy,f;

	// computing base values
	rx = first->x - second->x;
	ry = first->y - second->y;
	r = sqrt(rx*rx + ry*ry);

	if(r == 0.0)
		return;

	f = A / pow(r,6) - B / pow(r,12);
	fx = f * rx / r;
	fy = f * ry / r;

	// updating destination's structure
	second->fx = second->fx + fx;
	second->fy = second->fy + fy;

	// updating sources's structure
	first->fx = first->fx - fx;
	first->fy = first->fy - fy;
	
}

// Function for computing interaction among two particle arrays
void evolve(struct sharedParticle *first, struct sharedParticle *second, int limit){
	int j,k;
	
	for(j = 0; j < limit; j++){
		for(k = j+1; k < limit; k++){
			interact(&first[j],&second[k]);	
		}
	}
}

// Function to merge two particle arrays
// Permanent changes reside only in first array
void merge(struct sharedParticle *first, struct sharedParticle *second, int limit){
	int j;
	
	for(j = 0; j < limit; j++){
		first[j].fx += second[j].fx;
		first[j].fy += second[j].fy;
	}
}

// Function for updating velocity, acceleration and new position
void updateProperties(struct privateParticle *priv, struct sharedParticle *shared, int limit){
	int j;
	
	for(j = 0; j < limit; j++){
		priv[j].aX = shared[j].fx / MASS;
		priv[j].aY = shared[j].fy / MASS;
		priv[j].vX = priv[j].vX + priv[j].aX * DELTA;
		priv[j].vY = priv[j].vY + priv[j].aY * DELTA;
		shared[j].x = shared[j].x + priv[j].vX * DELTA;
		shared[j].y = shared[j].y + priv[j].vY * DELTA;
	}
}

