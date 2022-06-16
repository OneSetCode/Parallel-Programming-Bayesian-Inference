#include "matrices.h"
#include "regmodels.h"
#include "Rfunctions.h"
#include <mpi.h>
#include <iomanip>
#include <float.h>


// For MPI communication
#define BAYESLOG 1
#define SHUTDOWNTAG 0
// Used to determine PRIMARY or REPLICA
static int myrank;
// Global variables
int nobservations = 148;
int nvariables = 61;
gsl_matrix *data = gsl_matrix_alloc(nobservations, nvariables);
gsl_matrix *response = gsl_matrix_alloc(nobservations, 1);
// Function Declarations
void primary();
void replica(int primaryname);
void bayesLog(gsl_rng *mystream, double *result, int ind);

int main(int argc, char *argv[])
{
    FILE *dt;
    double tmp;
    // START THE MPI SESSION //
    MPI_Init(&argc, &argv);
    // What is the ID for the process? //
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Read in the data
    dt = fopen("534finalprojectdata.txt", "r");
    gsl_matrix_fscanf(dt, data);
    fclose(dt);
    // initialize the response vector
    for (int i = 0; i < nobservations; i++)
    {
        gsl_matrix_set(response, i, 0, gsl_matrix_get(data, i, 60));
    }
    // Branch off to primary or replica function
    // Primary has ID == 0, the replicas are then in order 1,2,3,...
    if (myrank == 0)
    {
        primary();
    }
    else
    {
        replica(myrank);
    }
    // clean memory
    gsl_matrix_free(data);
    gsl_matrix_free(response);
    // Finalize the MPI session
    MPI_Finalize();
    return (1);
}

// primary function
void primary()
{
    int rank;              // another looping variable
    int ntasks;            // the total number of replicas
    int jobsRunning;       // how many replicas we have working
    int work[1];           // information to send to the replicas
    double workresults[5]; // info received from the replicas
    FILE *fout;            // the output file
    MPI_Status status;     // MPI information
    int maxReg = 5;
    int A[nvariables - 1];
    gsl_matrix *beta = gsl_matrix_alloc(2, 1);
    LPRegression regressions = new Regression; // create a regression space
    regressions->Next = NULL;
    char resultfile[] = "result.txt";

    // Find out how many replicas there are
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    
    fprintf(stdout, "Total Number of processors = %d\n", ntasks);
    jobsRunning = 1;

    for (int var = 0; var < nvariables; var++)
    {
        // This will tell a replica which variable to work on
        work[0] = var;

        if (jobsRunning < ntasks) // Do we have an available processor?
        {
            // Send out a work request
            MPI_Send(&work,           // the vector with the variable
                     1,               // the size of the vector
                     MPI_INT,         // the type of the vector
                     jobsRunning,     // the ID of the replica to use
                     BAYESLOG,        // tells the replica what to do
                     MPI_COMM_WORLD); // send the request out to anyone
                                      // who is available
            printf("Primary sends out work request [%d] to replica [%d]\n", work[0], jobsRunning);
            // Increase the # of processors in use
            jobsRunning++;
        }
        else // all the processors are in use!
        {
            MPI_Recv(workresults, // where to store the results
                     5,           // the size of the vector
                     MPI_DOUBLE,  // the type of the vector
                     MPI_ANY_SOURCE,
                     MPI_ANY_TAG,
                     MPI_COMM_WORLD,
                     &status); // lets us know which processor
                               // returned these results
            printf("Primary has received the result of work request [%d] from replica [%d]\n",
                   (int)workresults[0], status.MPI_SOURCE);

            // Add the results to the regressions list
            A[0] = (int) workresults[0];
            gsl_matrix_set(beta, 0, 0, workresults[3]);
            gsl_matrix_set(beta, 1, 0, workresults[4]);
            AddRegression(maxReg, regressions, 1, A, beta, workresults[1], workresults[2]);

            printf("Primary sends out work request [%d] to replica [%d]\n",
                   work[0], status.MPI_SOURCE);
            // Send out a new work order to the processors that just
            // returned
            MPI_Send(&work,
                     1,
                     MPI_INT,
                     status.MPI_SOURCE, // the replica that just returned
                     BAYESLOG,
                     MPI_COMM_WORLD);
        } // using all the processors
    }     // loop over all the variables
    // loop over all the replicas
    for (rank = 1; rank < jobsRunning; rank++)
    {
        MPI_Recv(workresults,
                 5,
                 MPI_DOUBLE,
                 MPI_ANY_SOURCE, // whoever is ready to report back
                 MPI_ANY_TAG,
                 MPI_COMM_WORLD,
                 &status);
        printf("Primary has received the result of work request [%d]\n",
               (int)workresults[0]);

        A[0] = (int) workresults[0];
        gsl_matrix_set(beta, 0, 0, workresults[3]);
        gsl_matrix_set(beta, 1, 0, workresults[4]);
        AddRegression(maxReg, regressions, 1, A, beta, workresults[1], workresults[2]);
    }
    printf("Tell the replicas to shutdown.\n");
    // Shut down the replica processes
    for (rank = 1; rank < ntasks; rank++)
    {
        printf("Primary is shutting down replica [%d]\n", rank);
        MPI_Send(0,
                 0,
                 MPI_INT,
                 rank,        // shutdown this particular node
                 SHUTDOWNTAG, // tell it to shutdown
                 MPI_COMM_WORLD);
    }
    printf("got to the end of Primary code\n");

    SaveRegressions(resultfile, regressions);
    DeleteAllRegressions(regressions);
    gsl_matrix_free(beta);
    // return to the main function
    return;
}

void replica(int replicaname)
{
    int work[1];           // the input from primary
    double workresults[5]; // the output for primary
    MPI_Status status;     // for MPI communication
    const gsl_rng_type *TYPE;
    gsl_rng *RNG;
    gsl_rng_env_setup();
    TYPE = gsl_rng_default;
    RNG = gsl_rng_alloc(TYPE);
    gsl_rng_set(RNG, replicaname);
    // the replica listens for instructions...
    int notDone = 1;
    while (notDone)
    {
        printf("Replica %d is waiting\n", replicaname);
        MPI_Recv(&work,       // the input from primary
                 1,           // the size of the input
                 MPI_INT,     // the type of the input
                 0,           // from the PRIMARY node (rank=0)
                 MPI_ANY_TAG, // any type of order is fine
                 MPI_COMM_WORLD,
                 &status);
        printf("Replica %d just received smth\n", replicaname);
        // switch on the type of work request
        switch (status.MPI_TAG)
        {
        case BAYESLOG:
            printf("Replica %d has received work request [%d]\n", replicaname, work[0]);
            bayesLog(RNG, workresults, work[0]);
            // Send the results
            MPI_Send(&workresults,
                     5,
                     MPI_DOUBLE,
                     0, // send it to primary
                     0, // doesn't need a TAG
                     MPI_COMM_WORLD);
            printf("Replica %d finished processing work request [%d]\n", replicaname, work[0]);
            break;
        case SHUTDOWNTAG:
            printf("Replica %d was told to shutdown\n", replicaname);
            return;
        default:
            notDone = 0;
            printf("The replica code should never get here.\n");
            return;
        }
    }
    gsl_rng_free(RNG);
    return;
}


void bayesLog(gsl_rng *mystream, double *result, int ind)
{
    double MC, laplace, mean;
    gsl_matrix *obs = gsl_matrix_alloc(nobservations, 1);
    // set up the initial observation
    for (int i = 0; i < nobservations; i++)
    {
        gsl_matrix_set(obs, i, 0, gsl_matrix_get(data, i, ind));
    }
    // compute the betamode and posterior mean
    gsl_matrix *betaMode = getcoefNR(response, obs, nobservations, 1000);
    gsl_matrix *postMean = getPosteriorMeans(mystream, response, obs, betaMode, 10000, nobservations);
    // print out the result
    printf("The means are: \n");
    for (int j = 0; j < (postMean->size1); j++)
    {
        mean = gsl_matrix_get(postMean, j, 0);
        printf("beta%i = %.3f\n", j, mean);
    }
    //print out the Laplace approximations and monte carlo integration
    laplace = getLaplaceApprox(response, obs, betaMode, nobservations);
    printf("The Laplace approximation is %.3f \n", laplace);
    MC = log(getMonteCarlo(mystream, response, obs, 10000, nobservations));
    printf("The MC integration is %.3f \n", MC);
    //set the result
    result[0] = (double) (ind + 1);
    result[1] = MC;
    result[2] = laplace;
    result[3] = gsl_matrix_get(postMean, 0, 0);
    result[4] = gsl_matrix_get(postMean, 1, 0);
    //free the memory
    gsl_matrix_free(obs);
    gsl_matrix_free(betaMode);
    gsl_matrix_free(postMean);
}