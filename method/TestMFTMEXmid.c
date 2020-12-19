#include "mex.h"
#include <math.h>

/* Number of bins of the hash tables */
#define NUM_HASH_BINS 1024
/* Mask to compute hash value, must be log2(NUM_HASH_BINS) binary ones */
#define HASH_MASK 0x3FFu

/* 

Compute function approximation for the Discrete Mean Filter Transform. Version with mid sample quantiles.
Coded by Ezequiel Lopez-Rubio. July 2016.

In order to compile this MEX function, type the following at Matlab prompt:
>> mex TestMFTMEXmid.c

[TestFuncVal]=TestMFTMEXmid(Model,TestSamples);

Inputs:
	Model			MFT model
	TestSamples		DxM matrix with M test samples of dimension D
Output:
	TestFuncVal		Mx1 vector with the approximated function values

*/

typedef double elem_type ;

/*
 * QuickSort algorithm. Adapted from:
 * http://rosettacode.org/wiki/Sorting_algorithms/Quicksort#C
 *
 **/
void quick_sort (elem_type *a, int n) {
    int i, j;
    elem_type p, t;
    if (n < 2)
        return;
    p = a[n / 2];
    for (i = 0, j = n - 1;; i++, j--) {
        while (a[i] < p)
            i++;
        while (p < a[j])
            j--;
        if (i >= j)
            break;
        t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
    quick_sort(a, i);
    quick_sort(a + i, n - i);
}

/*
 *
 * Median based on the mid sample quantiles, see pages 229-230 in:
 * Asymptotic properties of sample quantiles of discrete distributions
 * Yanyuan Ma, Marc G. Genton, Emanuel Parzen
 * DOI: 10.1007/s10463-008-0215-z
 *
 * Note: This function can be used to compute mid sample quantiles other than
 * the median. This can be done by substituting while(mid_distrib_curr<=0.5)
 * by while(mid_distrib_curr<=quantile), where 0<quantile<1.
 *
 **/
elem_type quantile_median (elem_type *a, int n) {
    
    /* Previous and current values of the mid distribution function */
    double mid_distrib_prev,mid_distrib_curr;
    /* Previous and current values of the mid sample quantiles */
    elem_type quantile_prev,quantile_curr;
    /* Current value of the standard distribution function */
    double distrib_curr;
    /* Auxiliary variables */
    int cnt,cnt2;
    double one_over_n;
    
    /* Precompute 1/(number of samples)*/
    one_over_n=1.0f/n;
    
    /* Sort the input array */
    quick_sort(a,n);
    
    /* Initialize the previous values */
    mid_distrib_prev=0.0;
    quantile_prev=a[0];
    
    /* Initialize the current values with the first interval of equal values in the sorted array */
    cnt=1;
    while((cnt<n) && (a[cnt]==a[0]))
    {
        cnt++;
    }
    distrib_curr=cnt*one_over_n;
    mid_distrib_curr=0.5*cnt*one_over_n;
    quantile_curr=a[0];
    
    /* Look for the point where the mid distribution function exceeds 0.5 */
    while(mid_distrib_curr<=0.5)
    {
        /* Find the next interval of equal values in the sorted array */
        cnt2=cnt+1;
        while((cnt2<n) && (a[cnt2]==a[cnt]))
        {
            cnt2++;
        }

        /* Store the previous values of the mid distribution function and the mid quantile */
        mid_distrib_prev=mid_distrib_curr;
        quantile_prev=quantile_curr;
        
        /* Compute the new values of the mid distribution function,
        the standard distribution function, and the mid sample quantile. Please
        note that (cnt2-cnt)*one_over_n is the probability mass at value a[cnt]. */
        distrib_curr=cnt2*one_over_n;
        mid_distrib_curr=distrib_curr-0.5*(cnt2-cnt)*one_over_n;
        quantile_curr=a[cnt];
        
        /* Update the index over the array */
        cnt=cnt2;
    }
    
    /* Carry out the linear interpolation of the mid distribution function. A degenerate case
       is considered separately to avoid NaNs. */
    if (mid_distrib_curr>mid_distrib_prev)
    {
        return (quantile_prev+(quantile_curr-quantile_prev)*(0.5-mid_distrib_prev)/(mid_distrib_curr-mid_distrib_prev));
    }
    else
    {
        return 0.5*(quantile_curr+quantile_prev);
    }
    
}


/* Matrix sum. It supports that one of the operands is also the result*/
void MatrixSum(double *A,double *B,double *Result,int NumRows,int NumCols)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptra=A;
    ptrb=B;
    ptrres=Result;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=(*ptra)+(*ptrb);
        ptrres++;
        ptra++;
        ptrb++;
    }    
}

/* Matrix product */
void MatrixProduct(double *A,double *B,double *Result,int NumRowsA,
    int NumColsA,int NumColsB)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int i;
    register int j;
    register int k;
    register double Sum;
    
    ptrres=Result;
    for(j=0;j<NumColsB;j++)
    {
        for(i=0;i<NumRowsA;i++)
        {
            Sum=0.0;
            ptrb=B+NumColsA*j;
            ptra=A+i;
            for(k=0;k<NumColsA;k++)
            {
                Sum+=(*ptra)*(*ptrb);
                ptra+=NumRowsA;
                ptrb++;
            }    
            (*ptrres)=Sum;
            ptrres++;
        }
    }            
}   

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{

	mxArray *Model,*TestSamples,*MyMedianFilter,*MyIndices,*MyCounts,*MyFuncVals,*MyCell;
	int NumTestSamples,Dimension,NumMedianFilters;
	int NdxMedianFilter,NdxHash,NdxSample,NdxDim,MyNumElems;
	double *ptrTestSamples,*ptrTestFuncValues,*ptrMyHashBin;
	int *NumAveragedFilters;
	double *ptrA,*ptrb,*ptrMyA,*ptrMyb,*AuxVector,*AuxVector2,*FinalFuncVals;
	double **ptrIndices,**ptrCounts,**ptrFuncVals;
	double Factor;
	int *NumElemsHashBin;
	int HashValue,MyElement,NdxBin;


	/* Get input mxArrays */
	Model=prhs[0];
	TestSamples=prhs[1];

	/* Get working data */
	Dimension=mxGetM(TestSamples);
	NumTestSamples=mxGetN(TestSamples);
	ptrTestSamples=mxGetPr(TestSamples);
	NumMedianFilters=(int)mxGetScalar(mxGetField(Model,0,"NumMedianFilters"));
	ptrA=mxGetPr(mxGetField(Model,0,"A"));
	ptrb=mxGetPr(mxGetField(Model,0,"b"));

	/* Create output mxArray. Note that Matlab initializes its elements to zero. */
	plhs[0]=mxCreateDoubleMatrix(1,NumTestSamples,mxREAL);
	ptrTestFuncValues=mxGetPr(plhs[0]);

	/* Create auxiliary arrays */
	ptrIndices=mxMalloc(NUM_HASH_BINS*sizeof(double *));
	ptrCounts=mxMalloc(NUM_HASH_BINS*sizeof(double *));
	ptrFuncVals=mxMalloc(NUM_HASH_BINS*sizeof(double *));
	NumElemsHashBin=mxMalloc(NUM_HASH_BINS*sizeof(int));
	AuxVector=mxMalloc(Dimension*sizeof(double));
	AuxVector2=mxMalloc(Dimension*sizeof(double));
	NumAveragedFilters=mxCalloc(NumTestSamples,sizeof(int));
	FinalFuncVals=mxCalloc(NumTestSamples*NumMedianFilters,sizeof(double));

	/* For each median filter of the model */
	for(NdxMedianFilter=0;NdxMedianFilter<NumMedianFilters;NdxMedianFilter++)
	{
		ptrMyA=ptrA+Dimension*Dimension*NdxMedianFilter;
		ptrMyb=ptrb+Dimension*NdxMedianFilter;

		/* Obtain the pointers to the elements of the hash table */
		MyMedianFilter=mxGetCell(mxGetField(Model,0,"MedianFilter"),NdxMedianFilter);
		MyIndices=mxGetField(MyMedianFilter,0,"Indices");
		MyCounts=mxGetField(MyMedianFilter,0,"Counts");
		MyFuncVals=mxGetField(MyMedianFilter,0,"FuncValues");
		for(NdxHash=0;NdxHash<NUM_HASH_BINS;NdxHash++)
		{
			MyCell=mxGetCell(MyIndices,NdxHash);
			/* Check whether this element of the hash table is empty */
			if (MyCell==NULL)
			{
				ptrIndices[NdxHash]=NULL;
				ptrCounts[NdxHash]=NULL;
				NumElemsHashBin[NdxHash]=0;
			}
			else
			{
				ptrIndices[NdxHash]=mxGetPr(MyCell);
				NumElemsHashBin[NdxHash]=mxGetN(MyCell);
				MyCell=mxGetCell(MyCounts,NdxHash);
				ptrCounts[NdxHash]=mxGetPr(MyCell);
				MyCell=mxGetCell(MyFuncVals,NdxHash);
				ptrFuncVals[NdxHash]=mxGetPr(MyCell);				
			}
		}
		/* Process all test samples */
		for(NdxSample=0;NdxSample<NumTestSamples;NdxSample++)
		{
			/* Transform the sample */
			MatrixProduct(ptrMyA,ptrTestSamples+NdxSample*Dimension,
				AuxVector,Dimension,Dimension,1);
			MatrixSum(AuxVector,ptrMyb,AuxVector,Dimension,1);

			/* Round the result and compute hash value */
			HashValue=0;
			for(NdxDim=0;NdxDim<Dimension;NdxDim++)
			{
				AuxVector2[NdxDim]=floor(AuxVector[NdxDim]+0.5);
				HashValue+=(int)AuxVector2[NdxDim];
			}
			HashValue&=HASH_MASK;

			/* Look for the mean filter bin corresponding to this test sample */
			MyNumElems=NumElemsHashBin[HashValue];
			ptrMyHashBin=ptrIndices[HashValue];
			MyElement=-1;
			for(NdxBin=0;NdxBin<MyNumElems;NdxBin++)
			{
				if (memcmp(AuxVector2,ptrMyHashBin+NdxBin*Dimension,
					Dimension*sizeof(double))==0)
				{
					MyElement=NdxBin;
					break;
				}
			}

			/* If the mean filter bin has been found, add the corresponding function
			value to the list */
			if (MyElement>=0)
			{
				FinalFuncVals[NumAveragedFilters[NdxSample]+NdxSample*NumMedianFilters]=
					ptrFuncVals[HashValue][MyElement];
				NumAveragedFilters[NdxSample]++;
			}
		}
	}

	/* Compute the output as the median of the values coming from the individual median filters */
	for(NdxSample=0;NdxSample<NumTestSamples;NdxSample++)
	{
		ptrTestFuncValues[NdxSample]=
			quantile_median(FinalFuncVals+NdxSample*NumMedianFilters,
			NumAveragedFilters[NdxSample]);
	}

	/* Release dynamic memory */
	mxFree(ptrIndices);
	mxFree(ptrCounts);
	mxFree(ptrFuncVals);
	mxFree(AuxVector);
	mxFree(AuxVector2);
	mxFree(NumElemsHashBin);
	mxFree(NumAveragedFilters);
	mxFree(FinalFuncVals);

}