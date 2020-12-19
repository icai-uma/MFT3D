#include "mex.h"
#include <math.h>

/* Number of bins of the hash tables */
#define NUM_HASH_BINS 1024
/* Mask to compute hash value, must be log2(NUM_HASH_BINS) binary ones */
#define HASH_MASK 0x3FFu
/* Define maximum number of elements per hash bin table chunk */
#define ELEMS_PER_HASH_BIN_CHUNK 512

/* 

Compute the medians for the Median Histogram Transform. Version with mid sample quantiles.
Coded by Ezequiel Lopez-Rubio. July 2016.

In order to compile this MEX function, type the following at Matlab prompt:
>> mex ComputeMediansMEXmid.c

[Medians]=ComputeMediansMEXmid(Samples,FuncValues,A,b);

Inputs:
	Samples		DxN matrix with N training samples of dimension D
	FuncValues	1xN matrix with N function values corresponding to Samples
	A			DxD matrix A of an affine transform
	b			Dx1 vector b of an affine transform
Output:
	Medians	Resulting medians structure

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

	mxArray *Samples,*MyCell,*Indices,*Counters,*FuncValues;
	int NumSamples,Dimension;
	int NdxSample,NdxDim,MyNumElems;
	double *ptrSamples,*ptrFuncValues,*ptrMyHashBin,*ptrComputedMedians;
	double *TempFuncVals;
	int *ptrNextFuncVal;
	double *ptrA,*ptrb,*AuxVector,*AuxVectorRounded;
	double **ptrIndices,**ptrCounts;
	int **ptrFirstFuncVals;
	int *NumElemsHashBin;
	int HashValue,MyElement,NdxBin,NewTableSize,NdxElemHashBin,LinkedListSize;
	int NdxFuncVal,CurrFuncVal,TempFuncValsSize,NewTempFuncValsSize;
	const char *FieldNames[]={"Indices","Counts","FuncValues"};


	/* Get input mxArrays */
	Samples=prhs[0];
	ptrFuncValues=mxGetPr(prhs[1]);
	ptrA=mxGetPr(prhs[2]);
	ptrb=mxGetPr(prhs[3]);

	/* Get working data */
	Dimension=mxGetM(Samples);
	NumSamples=mxGetN(Samples);
	ptrSamples=mxGetPr(Samples);

	
	/* Create auxiliary arrays */
	ptrIndices=mxCalloc(NUM_HASH_BINS,sizeof(double *)); /*Aloja en memoria 1024 elementos de tamaño puntero a double*/
	ptrCounts=mxCalloc(NUM_HASH_BINS,sizeof(double *));
	ptrFirstFuncVals=mxCalloc(NUM_HASH_BINS,sizeof(int *));
	ptrNextFuncVal=mxMalloc(NumSamples*sizeof(int));
	NumElemsHashBin=mxCalloc(NUM_HASH_BINS,sizeof(int));
	AuxVector=mxMalloc(Dimension*sizeof(double));
	AuxVectorRounded=mxMalloc(Dimension*sizeof(double));
    /*Calloc pone a cero la memoria reservada pero Malloc no*/

	/* Process all input samples */
	for(NdxSample=0;NdxSample<NumSamples;NdxSample++)
	{
		/* Transform the sample */
		MatrixProduct(ptrA,ptrSamples+NdxSample*Dimension,
			AuxVector,Dimension,Dimension,1);   /*A*x, x ya es alfa*x porque se ha aplicado el zoomfactor antes*/
		MatrixSum(AuxVector,ptrb,AuxVector,Dimension,1); /*A*x+b*/

		/* Round the result and compute hash value */
		HashValue=0;
		for(NdxDim=0;NdxDim<Dimension;NdxDim++)
		{
			AuxVectorRounded[NdxDim]=floor(AuxVector[NdxDim]+0.5);
			HashValue+=(int)AuxVectorRounded[NdxDim];   /*Pueden repetirse*/
		}
		HashValue&=HASH_MASK;

		/* Look for the median filter bin corresponding to this test sample */
		MyNumElems=NumElemsHashBin[HashValue];
		ptrMyHashBin=ptrIndices[HashValue];
		MyElement=-1;
		for(NdxBin=0;NdxBin<MyNumElems;NdxBin++)
		{
			if (memcmp(AuxVectorRounded,ptrMyHashBin+NdxBin*Dimension,
				Dimension*sizeof(double))==0)
			{
				MyElement=NdxBin;
				break;
			}
		}

		/* If the median filter bin has been found, add one to the counter and link
		   the function value to the existing linked list.
		   Otherwise, insert a new median filter bin into the hash table and create
		   a linked list of function values with only one element. 
		   Please note that if the corresponding hash table bin was empty,
		   then a large chunk of memory is allocated so that less memory allocations
		   are needed. */
		if (MyElement>=0)
		{
			/* Median filter bin found */
			ptrCounts[HashValue][MyElement]++;
			ptrNextFuncVal[NdxSample]=ptrFirstFuncVals[HashValue][MyElement];
			ptrFirstFuncVals[HashValue][MyElement]=NdxSample;
		}
		else
		{
			/* Median filter bin not found */
			MyNumElems++;
			NumElemsHashBin[HashValue]=MyNumElems;
			if ((MyNumElems%ELEMS_PER_HASH_BIN_CHUNK)==1)
			{
				NewTableSize=ELEMS_PER_HASH_BIN_CHUNK*((MyNumElems/ELEMS_PER_HASH_BIN_CHUNK)+1);
				ptrCounts[HashValue]=mxRealloc(ptrCounts[HashValue],NewTableSize*sizeof(double));
				ptrIndices[HashValue]=mxRealloc(ptrIndices[HashValue],NewTableSize*Dimension*sizeof(double));
				ptrFirstFuncVals[HashValue]=mxRealloc(ptrFirstFuncVals[HashValue],NewTableSize*sizeof(double));
			}
			ptrCounts[HashValue][MyNumElems-1]=1;   /*Nº de muestras dentro de un paralelepípedo?*/
			ptrFirstFuncVals[HashValue][MyNumElems-1]=NdxSample;
			ptrNextFuncVal[NdxSample]=-1; /* This is to mark the end of the linked list of function values */
			memcpy(ptrIndices[HashValue]+(MyNumElems-1)*Dimension,
				AuxVectorRounded,Dimension*sizeof(double));
		}
	}


	/* Convert the hash table to mxArray format and compute the median values */
	Indices=mxCreateCellMatrix(NUM_HASH_BINS, 1);
	Counters=mxCreateCellMatrix(NUM_HASH_BINS, 1);
	FuncValues=mxCreateCellMatrix(NUM_HASH_BINS, 1);
	for(NdxBin=0;NdxBin<NUM_HASH_BINS;NdxBin++)
	{
		if (NumElemsHashBin[NdxBin]>0)
		{
			/* Indices array */
			MyCell=mxCreateDoubleMatrix(Dimension,NumElemsHashBin[NdxBin],mxREAL);
			memcpy(mxGetPr(MyCell),ptrIndices[NdxBin],
				NumElemsHashBin[NdxBin]*Dimension*sizeof(double));
			mxSetCell(Indices,NdxBin,MyCell);
			/* Counters array */
			MyCell=mxCreateDoubleMatrix(1,NumElemsHashBin[NdxBin],mxREAL);
			memcpy(mxGetPr(MyCell),ptrCounts[NdxBin],
				NumElemsHashBin[NdxBin]*sizeof(double));
			mxSetCell(Counters,NdxBin,MyCell);
			/* Function values array */
			MyCell=mxCreateDoubleMatrix(1,NumElemsHashBin[NdxBin],mxREAL);
			ptrComputedMedians=mxGetPr(MyCell);
			TempFuncVals=mxMalloc(ELEMS_PER_HASH_BIN_CHUNK*sizeof(double));
			TempFuncValsSize=ELEMS_PER_HASH_BIN_CHUNK;
			for(NdxElemHashBin=0;NdxElemHashBin<NumElemsHashBin[NdxBin];NdxElemHashBin++)
			{
				/* Enlarge the auxiliary array to hold the function values if necessary */
				if (ptrCounts[NdxBin][NdxElemHashBin]>TempFuncValsSize)
				{
					NewTempFuncValsSize=((((int)ptrCounts[NdxBin][NdxElemHashBin])/ELEMS_PER_HASH_BIN_CHUNK)+1)
						*ELEMS_PER_HASH_BIN_CHUNK;
					TempFuncVals=mxRealloc(TempFuncVals,NewTempFuncValsSize*sizeof(double));
					TempFuncValsSize=NewTempFuncValsSize;
				}
				/* Traverse the linked list */
				CurrFuncVal=ptrFirstFuncVals[NdxBin][NdxElemHashBin];
				LinkedListSize=ptrCounts[NdxBin][NdxElemHashBin];
				for(NdxFuncVal=0;NdxFuncVal<LinkedListSize;NdxFuncVal++)
				{
					TempFuncVals[NdxFuncVal]=ptrFuncValues[CurrFuncVal];
					CurrFuncVal=ptrNextFuncVal[CurrFuncVal];
				}
				ptrComputedMedians[NdxElemHashBin]=quantile_median(TempFuncVals,LinkedListSize);
			}
			mxSetCell(FuncValues,NdxBin,MyCell);
		}
	}


	/* Create output mxArray */
	plhs[0]=mxCreateStructMatrix(1, 1, 3, FieldNames);
	mxSetField(plhs[0], 0, "Indices", Indices);
	mxSetField(plhs[0], 0, "Counts", Counters);
	mxSetField(plhs[0], 0, "FuncValues", FuncValues);

	/* Release dynamic memory */
	mxFree(AuxVector);
	mxFree(AuxVectorRounded);
	mxFree(NumElemsHashBin);
	for(NdxBin=0;NdxBin<NUM_HASH_BINS;NdxBin++)
	{
		mxFree(ptrIndices[NdxBin]);
		mxFree(ptrCounts[NdxBin]);
		mxFree(ptrFirstFuncVals[NdxBin]);
	}
	mxFree(ptrIndices);
	mxFree(ptrCounts);
	mxFree(ptrFirstFuncVals);
	mxFree(ptrNextFuncVal);
}