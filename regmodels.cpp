/*
 FILE: REGMODELS.CPP
*/

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "regmodels.h"


//tests if two vectors are equal
//we assume that the two vectors are sorted
int sameregression(int lenA1,int* A1,int lenA2,int* A2)
{
  int i;

  if(lenA1!=lenA2)
  {
    return 0;
  }

  //the two vectors have the same length
  //are their elements equal?
  for(i=0;i<lenA1;i++)
  {
     if(A1[i]!=A2[i])
     {
       return 0;
     }
  }

  return 1;
}

// this function checks whether our new regression is in the top "nMaxRegs"
// based on marginal likelihood. If the new model is better than any of the
// existing regressions in the list, it is added to the list
// and the worst regression is discarded. Here "regressions" represents
// the head of the list, "lenA" is the number of predictors
// and "logmarglikA" is the marginal likelihood of the regression
// with predictors A.
void AddRegression(int nMaxRegs, LPRegression regressions, int lenA, int* A, gsl_matrix* beta, double MC, double laplace)
{
  int i, j = 0;

  LPRegression p = regressions;
  LPRegression pnext = p->Next;

  while(NULL != pnext && j < nMaxRegs)
  {
     //return if we have previously found this regression
     if(sameregression(lenA, A, pnext->lenA, pnext->A))
     {
        return;
     }

     //go to the next element in the list if the current
     //regression has a larger log marginal likelihood than
     //the new regression A
     if(pnext->MC > MC)
     {
        p = pnext;
        pnext = p->Next;
     }
     else //otherwise stop; this is where we insert the new regression
     {
        break;
     }
     j++;
  }

  // if we reached "nMaxRegs" we did not beat any of the top 10 with the new regression.
  // Otherwise we add it like normal.

  if(nMaxRegs == j)
  {
	  return;
  }

  //create a new element of the list
  LPRegression newp = new Regression;
  newp->lenA = lenA;
  newp->MC = MC;
  newp->laplace = laplace;
  newp->A = new int[lenA];
  newp->beta = new double[lenA + 1];
  
  //copy the predictors
  for(i=0;i<lenA;i++)
  {
    newp->A[i] = A[i];
  }

  for(i = 0; i < lenA + 1; i++) 
  {
    newp->beta[i] = gsl_matrix_get(beta, i, 0);
  }

  //insert the new element in the list
  p->Next = newp;
  newp->Next = pnext;

  // now we move through the list until we either reach the end of it, or reach the
  // element just after the "nMaxRegs" element.
  while(j < nMaxRegs && NULL!=pnext)
  {
	  p = pnext;
	  pnext = p->Next;
	  j++;
  }
  // if we reach nMaxRegs, we have to discard the new worst element in the list.
  if(nMaxRegs == j)
  {
	  DeleteLastRegression(regressions);
  }
  return;
}

//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPRegression regressions)
{
  //this is the first regression
  LPRegression p = regressions->Next;
  LPRegression pnext;

  while(NULL!=p)
  {
    //save the link to the next element of p
    pnext = p->Next;

    //delete the element specified by p
    //first free the memory of the vector of regressors
    delete[] p->A;
    p->Next = NULL;
    delete p;

    //move to the next element
    p = pnext;
  }

  return;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
  //this is the element before the first regression
  LPRegression pprev = regressions;
  //this is the first regression
  LPRegression p = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p)
  {
     return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL!=p->Next)
  {
    pprev = p;
    p = p->Next;
  }
  
  //now "p" should give the last element
  //delete it
  delete[] p->A;
  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}

//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPRegression regressions)
{
  int i;
  //open the output file
  FILE* out = fopen(filename,"w");
	
  if(NULL==out)
  {
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  //this is the first regression
  LPRegression p = regressions->Next;
  while(NULL!=p)
  {
    for(i=0;i<p->lenA;i++)
    {
       fprintf(out,"\t%i",p->A[i]);
    }
    //print the log marginal likelhood and the number of predictors
    fprintf(out,"%.3f\t%.3f",p->MC, p->laplace);
    //now save the predictors
    for(i=0;i<p->lenA + 1;i++)
    {
       fprintf(out,"\t%.3f",p->beta[i]);
    }
    fprintf(out,"\n");

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}

