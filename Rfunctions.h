#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

//THESE ARE GSL FUNCTIONS
//YOU DO NOT NEED TO INCLUDE ALL THESE HEADER FILES IN YOUR CODE
//JUST THE ONES YOU ACTUALLY NEED;
//IN THIS APPLICATION, WE ONLY NEED gsl/gsl_matrix.h
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_errno.h>



double inverseLogit(double x);
double inverseLogit2(double x);
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta, int n);
gsl_matrix* getPi2(gsl_matrix* x, gsl_matrix* beta, int n);
double logisticLogLik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, int n);
double lStar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, int n);
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* grad, int n);
void getHessian(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* Hess, int n);
gsl_matrix* getcoefNR(gsl_matrix* y, gsl_matrix* x, int n, int max);
gsl_matrix* MHlogistic(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int iter, int n);
double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int n);
double getMonteCarlo(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, int size, int n);
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int iter, int n);
gsl_matrix* makeCholesky(gsl_matrix* K);
void randomMVN(gsl_rng* mystream, gsl_matrix* Samples, gsl_matrix* mean, gsl_matrix* Sigma);