#include "Rfunctions.h"
#include "matrices.h"
#include "regmodels.h"


//compute the inverse of logit function
double inverseLogit(double x)
{
    double inverse = exp(x) / (exp(x) + 1.0);
    return(inverse);
}

double inverseLogit2(double x) 
{
    double inverse = exp(x) / pow(exp(x) + 1.0, 2.0);
	return(inverse);
}

//compute Pi_i
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta, int n) 
{
	gsl_matrix* result = gsl_matrix_alloc(n, 1);
    gsl_matrix* x0 = gsl_matrix_alloc(n, 2);
	//set x0
    for(int i = 0; i < n; i++) 
    {
		gsl_matrix_set(x0, i, 0, 1.0);
		gsl_matrix_set(x0, i, 1, gsl_matrix_get(x, i, 0)); 
	}
	//compute the multipy of x0 and beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, result);
	//compute the inverse logit function
	for(int j = 0; j < n; j++) 
    {
        double inverseLog = inverseLogit(gsl_matrix_get(result, j, 0));
		gsl_matrix_set(result, j, 0, inverseLog);
	}
    //free the memory
	gsl_matrix_free(x0);

	return(result);
}

//#another function for the computation of the Hessian
gsl_matrix* getPi2(gsl_matrix* x, gsl_matrix* beta, int n) 
{
    gsl_matrix* result = gsl_matrix_alloc(n, 1);
	gsl_matrix* x0 = gsl_matrix_alloc(n, 2);
	//compute x0
	for(int i = 0; i < n; i++) 
    {
		gsl_matrix_set(x0, i, 0, 1.0); 
		gsl_matrix_set(x0, i, 1, gsl_matrix_get(x, i, 0)); 
	}
	//compute the multipy of x0 and beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, result);
	//compute the inverse Logit
	for(int j = 0; j < n; j++) 
    {
        double inverseLog2 = inverseLogit2(gsl_matrix_get(result, j, 0));
		gsl_matrix_set(result, j, 0, inverseLog2);
	}
    //free the memory
	gsl_matrix_free(x0);
    
	return(result);
}

//#logistic log-likelihood
double logisticLogLik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, int n) 
{
	gsl_matrix* Pi = getPi(x, beta, n);
    //compute the log-like function
    double pi;
	double yi;
    double result = 0;
	for(int i = 0; i < n; i++) 
    {
		yi = gsl_matrix_get(y, i, 0);
		pi = gsl_matrix_get(Pi, i, 0);
		result += yi * log(pi) + (1.0 - yi) * log(1.0 - pi);
	}
	gsl_matrix_free(Pi);
	return(result);
}

//#calculates l^*(\beta_0,\beta_1)
double lStar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, int n) 
{
    int bsize1 = beta -> size1;
    int bsize2 = beta -> size2;
    gsl_matrix* Pi = getPi(x, beta, n);
	double yi, pi, result;
    double logLogLike = 0;
	double sum = 0;
    //compute logLogLike
	for(int i = 0; i < n; i++) 
    {
		yi = gsl_matrix_get(y, i, 0);
		pi = gsl_matrix_get(Pi, i, 0);
		logLogLike += yi * log(pi) + (1.0 - yi) * log(1.0 - pi);
	}
	for(int i = 0; i < bsize1; i++) 
    {
		for(int j = 0; j < bsize2; j++) 
        {
			sum += pow(gsl_matrix_get(beta, i, j), 2.0);
		}
	}
    //compute the result
	result = -log(2.0 * M_PI) - 0.5 * sum + logLogLike;

	gsl_matrix_free(Pi);
	return(result);
}


//compute or update the gradient    
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* grad, int n) 
{
	double g1 = 0, g2 = 0;
    double xi, yi, pi;
	// Get Pi
	gsl_matrix* Pi = getPi(x, beta, n);

	for(int i = 0; i < n; i++) 
    {
		xi = gsl_matrix_get(x, i, 0);
		yi = gsl_matrix_get(y, i, 0);
        pi = gsl_matrix_get(Pi, i, 0);
		g1 += (yi - pi);
		g2 += (yi - pi) * xi;
	}
	g1 -= gsl_matrix_get(beta, 0, 0);
	g2 -= gsl_matrix_get(beta, 1, 0);
    //set the gradient
	gsl_matrix_set(grad, 0, 0, g1);
	gsl_matrix_set(grad, 1, 0, g2);
    //free the memory
	gsl_matrix_free(Pi);
}


//compute Hesian
void getHessian(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* Hess, int n) 
{
	double h1 = 0, h2 = 0, h3 = 0, h4 = 0;
    double xi, pi;
	//compute Pi2
	gsl_matrix* Pi2 = getPi2(x, beta, n);

	for(int i = 0; i < n; i++) 
    {
        xi = gsl_matrix_get(x, i, 0);
		pi = gsl_matrix_get(Pi2, i, 0);
		h1 += pi;
		h2 += pi * xi;
		h4 += pi * pow(xi, 2.0);
	}
	h1 += 1.0;
	h3 = h2;
	h4 += 1.0;
    //set the hessian matrix
	gsl_matrix_set(Hess, 0, 0, -h1);
    gsl_matrix_set(Hess, 0, 1, -h2);
	gsl_matrix_set(Hess, 1, 0, -h3);
	gsl_matrix_set(Hess, 1, 1, -h4);
    //free memory
	gsl_matrix_free(Pi2);
}

//#this function implements our own Newton-Raphson procedure
gsl_matrix* getcoefNR(gsl_matrix* y, gsl_matrix* x, int n, int max) 
{
	double oldLog, newLog;
	//set the beta matrix
	gsl_matrix* beta = gsl_matrix_alloc(2, 1);
	gsl_matrix_set_zero(beta);
	gsl_matrix* nextBeta = gsl_matrix_alloc(2, 1);
    //set the needed matries
	gsl_matrix* Hess = gsl_matrix_alloc(2, 2);
	gsl_matrix* Grad = gsl_matrix_alloc(2, 1);
	gsl_matrix* inverHess = gsl_matrix_alloc(2, 2);
	gsl_matrix* hessGrad = gsl_matrix_alloc(2, 1);
    //compute the l* function
	oldLog = lStar(y, x, beta, n);
    //start the iteration
    for(int i = 0; i < max; i++)
    {
		getHessian(x, beta, Hess, n);
		getGradient(y, x, beta, Grad, n);
		inverHess = inverse(Hess);
        //matrix multiply
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inverHess, Grad, 0.0, hessGrad);
        //set new beta
		gsl_matrix_memcpy(nextBeta, beta);
		gsl_matrix_sub(nextBeta, hessGrad);	
		newLog = lStar(y, x, nextBeta, n);
        //see if any progress has been made
		if(newLog < oldLog) 
        {
			printf("CODING ERROR!!");
			exit(1);
		}
		//update the beta
		gsl_matrix_memcpy(beta, nextBeta);	
		//see if it is time to break
		if((newLog - oldLog) < 0.00001) 
        {
			break;
		} 
        else 
        {
			oldLog = newLog;
		}
	}
	//free the memory
	gsl_matrix_free(nextBeta);
	gsl_matrix_free(Hess);
	gsl_matrix_free(Grad);
	gsl_matrix_free(inverHess);
	gsl_matrix_free(hessGrad);

	return(beta);
}

//implement the MH algorithm
gsl_matrix* MHlogistic(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int iter, int n) 
{
	double oldLog, newLog;
    //set sample matrix
	gsl_matrix* sample = gsl_matrix_alloc(iter, 2);
	gsl_matrix_set_zero(sample);
    //set hessian matrix
	gsl_matrix* Hess = gsl_matrix_alloc(2, 2);
	getHessian(x, betaMode, Hess, n);
    //set the sigma
	gsl_matrix* sigma = inverse(Hess);
	gsl_matrix_scale(sigma, -1.0);
    //set the old and new beta
	gsl_matrix* oldBeta = gsl_matrix_alloc(2, 1);
	gsl_matrix_memcpy(oldBeta, betaMode);
	gsl_matrix* newBeta = gsl_matrix_alloc(2, 1);

	for(int i = 0; i < iter; i++) 
    {
		randomMVN(mystream, newBeta, oldBeta, sigma);
		//compute the old and new Loglik
		newLog = lStar(y, x, newBeta, n); 
        oldLog = lStar(y, x, oldBeta, n);
        //decide if there is big enough progress
		if(newLog - oldLog >= 1) 
        {
			gsl_matrix_memcpy(oldBeta, newBeta);
		} 
        else 
        {
			double u = gsl_ran_flat(mystream, 0.0, 1.0);
			if(log(u) <= newLog - oldLog) 
            {
				gsl_matrix_memcpy(oldBeta, newBeta);
			}
		}
		gsl_matrix_set(sample, i, 0, gsl_matrix_get(oldBeta, 0, 0));
		gsl_matrix_set(sample, i, 1, gsl_matrix_get(oldBeta, 1, 0));
	}
	//free the memory
	gsl_matrix_free(Hess);
	gsl_matrix_free(sigma);
	gsl_matrix_free(oldBeta);
	gsl_matrix_free(newBeta);

	return(sample);
}


// approximate the marginal likelihood using the Laplace approximation
double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int n) 
{
	double lstar, result;
	//compute the hessian matrix
	gsl_matrix* Hess = gsl_matrix_alloc(2,2);
	getHessian(x, betaMode, Hess, n);
	gsl_matrix_scale(Hess, -1.0);
	//compute l*
	lstar = lStar(y, x, betaMode, n);
	//compute the loglik
	result = log(2.0 * M_PI) + lstar - 0.5 * logdet(Hess);
	// Free memory
	gsl_matrix_free(Hess);

	return(result);
}

//calculate the marginal likelihood using Monte Carlo approximation
double getMonteCarlo(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, int size, int n) 
{
	double result, sum;
	//set the mean and sigma and beta
	gsl_matrix* mean = gsl_matrix_alloc(2, 1);
	gsl_matrix_set_zero(mean);
	gsl_matrix* sigma = gsl_matrix_alloc(2, 2);
	gsl_matrix_set_identity(sigma);
	gsl_matrix* beta = gsl_matrix_alloc(2, 1);
    //start the sample
    sum = 0;
	for(int i = 0; i < size; i++) 
    {
		randomMVN(mystream, beta, mean, sigma);
		sum += exp(logisticLogLik(y, x, beta, n));
	}
    //compute the result
	result = sum / (double) size;
	//free the memory
	gsl_matrix_free(mean);
	gsl_matrix_free(sigma);
	gsl_matrix_free(beta);

	return(result);
}

//compute the post mean
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int iter, int n) 
{
	gsl_vector_view c;
	//simulate the sample matrix and set the mean matrix
	gsl_matrix* sample = MHlogistic(mystream, y, x, betaMode, iter, n);
	gsl_matrix* mean = gsl_matrix_alloc(2, 1);
	int p = sample -> size1;
    int k = sample -> size2;
    //start iteration
	for(int i = 0; i < k; i++) 
    {
		c = gsl_matrix_column(sample, i);
		gsl_matrix_set(mean, i, 0, gsl_stats_mean(c.vector.data, c.vector.stride, p));
	}
    //free the memory
	gsl_matrix_free(sample);

	return(mean);
}


//creates the Cholesky decomposition of a matrix
gsl_matrix* makeCholesky(gsl_matrix* K)
{
	int i,j;
	
	gsl_matrix* Phi = gsl_matrix_alloc(K->size1,K->size1);
	if(GSL_SUCCESS!=gsl_matrix_memcpy(Phi,K))
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}
	if(GSL_SUCCESS!=gsl_linalg_cholesky_decomp(Phi))
	{
		printf("GSL failed Cholesky decomposition.\n");
		exit(1);
	}
	for(i=0;i<Phi->size1;i++)
	{
		for(j=i+1;j<Phi->size2;j++)
		{
			gsl_matrix_set(Phi,i,j,0.0);
		}
	}
	return(Phi);
}

//samples from the multivariate normal distribution N(mean,Sigma)
//the samples are saved in the matrix "Samples"
void randomMVN(gsl_rng* mystream, gsl_matrix* Samples, gsl_matrix* mean, gsl_matrix* Sigma)
{
  gsl_matrix* Psi = makeCholesky(Sigma);
  gsl_matrix* Z = gsl_matrix_alloc(Sigma->size1,1);
  gsl_matrix* X = gsl_matrix_alloc(Sigma->size1,1);
  gsl_vector* V = gsl_vector_alloc(Sigma->size1);
  for(int asample=0;asample<Samples->size2;asample++)
  {
    for(int i = 0; i < Sigma->size1; i++)
    {
      gsl_matrix_set(Z,i,0,gsl_ran_ugaussian(mystream));
    }
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
		    1.0, Psi, Z,
		    0.0, X);
    //add the mean
    gsl_matrix_add(X, mean);
    //record the sample we just generated
    gsl_matrix_get_col(V, X, 0);
	gsl_matrix_set_col(Samples, asample, V);
  }
  //free memory
  gsl_matrix_free(Psi);
  gsl_matrix_free(Z);
  gsl_matrix_free(X);
  gsl_vector_free(V);
  return;
}
