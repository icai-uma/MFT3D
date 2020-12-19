#include "mex.h"
#include <math.h>

/* Number of bins of the hash tables */
#define NUM_HASH_BINS 1024
/* Mask to compute hash value, must be log2(NUM_HASH_BINS) binary ones */
#define HASH_MASK 0x3FFu

/* 

Compute function approximation for the Discrete Mean Filter Transform.
Coded by Ezequiel Lopez-Rubio. October 2014.

In order to compile this MEX function, type the following at Matlab prompt:
>> mex TestMFTMEX.c

[TestFuncVal]=TestMFTMEX(Model,TestSamples);

Inputs:
	Model			MFT model
	TestSamples		DxM matrix with M test samples of dimension D
Output:
	TestFuncVal		Mx1 vector with the approximated function values

*/


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
	double *ptrTestSamples,*ptrTestFuncValues,*ptrMyHashBin,*NumAveragedFilters;
	double *ptrA,*ptrb,*ptrMyA,*ptrMyb,*AuxVector,*AuxVector2;
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
	NumAveragedFilters=mxCalloc(NumTestSamples,sizeof(double));

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
			value */
			if (MyElement>=0)
			{
				ptrTestFuncValues[NdxSample]+=ptrFuncVals[HashValue][MyElement];
				NumAveragedFilters[NdxSample]++;
			}
		}
	}

	/* Normalize the function value of all test samples */
	for(NdxSample=0;NdxSample<NumTestSamples;NdxSample++)
	{
		/* Normalize and add Gaussian density */
		ptrTestFuncValues[NdxSample]=
			ptrTestFuncValues[NdxSample]/NumAveragedFilters[NdxSample];
	}

	/* Release dynamic memory */
	mxFree(ptrIndices);
	mxFree(ptrCounts);
	mxFree(ptrFuncVals);
	mxFree(AuxVector);
	mxFree(AuxVector2);
	mxFree(NumElemsHashBin);
	mxFree(NumAveragedFilters);

}