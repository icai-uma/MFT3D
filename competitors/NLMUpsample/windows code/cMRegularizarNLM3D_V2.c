/**************************************************************************
%
%   Jose V. Manjon - jmanjon@fis.upv.es                                     
%   Universidad Politecinca de Valencia, Spain                               
%   Pierrick Coupe - pierrick.coupe@gmail.com                               
%   Brain Imaging Center, Montreal Neurological Institute.                  
%   Mc Gill University                                                      
%                                                                         
%   Copyright (C) 2010 Jose V. Manjon and Pierrick Coupe                    
%
%
**************************************************************************/

#include "math.h"
#include "mex.h"
#include <stdlib.h>
#include "matrix.h"
// undef needed for LCC compiler
#undef EXTERN_C
#include <windows.h>
#include <process.h>  

typedef struct{
    int rows;
    int cols;
    int slices;
    double * in_image;
    double * out_image;
    double * mean_image;
    double * pesos;    
    int ini;
    int fin;
    int radio;
    int f;
    //int th;
    double * sigma;
}myargument;


double distancia(double* ima,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz)
{
double d,acu,distancetotal,inc;
int i,j,k,ni1,nj1,ni2,nj2,nk1,nk2;

distancetotal=0;

for(k=-f;k<=f;k++)
{
 nk1=z+k;
 nk2=nz+k;  
 if(nk1<0) nk1=-nk1;
 if(nk2<0) nk2=-nk2;
 if(nk1>=sz) nk1=2*sz-nk1-1;    
 if(nk2>=sz) nk2=2*sz-nk2-1;

 for(j=-f;j<=f;j++)
 {
  nj1=y+j;
  nj2=ny+j;    
  if(nj1<0) nj1=-nj1;    
  if(nj2<0) nj2=-nj2;
  if(nj1>=sy) nj1=2*sy-nj1-1;
  if(nj2>=sy) nj2=2*sy-nj2-1;

  for(i=-f;i<=f;i++)
  {
    ni1=x+i;
    ni2=nx+i;    
    if(ni1<0) ni1=-ni1;
    if(ni2<0) ni2=-ni2;    
	if(ni1>=sx) ni1=2*sx-ni1-1;
    if(ni2>=sx) ni2=2*sx-ni2-1;
                
    distancetotal = distancetotal + ((ima[nk1*(sx*sy)+(nj1*sx)+ni1]-ima[nk2*(sx*sy)+(nj2*sx)+ni2])*(ima[nk1*(sx*sy)+(nj1*sx)+ni1]-ima[nk2*(sx*sy)+(nj2*sx)+ni2]));   
  }
 }
}

acu=(2*f+1)*(2*f+1)*(2*f+1);
d=distancetotal/acu;

return d;

}

unsigned __stdcall ThreadFunc( void* pArguments )
{
    double *ima,*fima,*medias,*pesos,w,d,*h,t1;
    int ii,jj,kk,ni,nj,nk,i,j,k,ini,fin,rows,cols,slices,v,p,p1,f,rc;

    myargument arg;
    arg=*(myargument *) pArguments;

    rows=arg.rows;    
    cols=arg.cols;
    slices=arg.slices;
    ini=arg.ini;    
    fin=arg.fin;
    ima=arg.in_image;
    fima=arg.out_image;
    medias=arg.mean_image;
    pesos=arg.pesos;
    v=arg.radio;
    f=arg.f;
    //th=arg.th;
    h=arg.sigma;    
    rc=rows*cols;
    
    /* filter*/
    for(k=ini;k<fin;k++)
    {
	for(j=0;j<rows;j++)
	{
		for(i=0;i<cols;i++)
		{		
            p=k*rc+j*cols+i;
            if(h[p]<1) continue;
            
			for(kk=0;kk<=v;kk++)
            {
              nk=k+kk;                
			  for(ii=-v;ii<=v;ii++)
			  {
              ni=i+ii;			
			  for(jj=-v;jj<=v;jj++)
			  {
				nj=j+jj;
				
                if(kk==0 && jj<0) continue;             
                if(kk==0 && jj==0 && ii<=0) continue;  
				
				if(ni>=0 && nj>=0 && nk>=0 && ni<cols && nj<rows && nk<slices)
				{
                     p1=nk*rc+nj*cols+ni;
                                          
                     t1 = abs(medias[p]-medias[p1]);  
              
                     if(t1>0.6*h[p]) continue;
                      
                     d=distancia(ima,i,j,k,ni,nj,nk,f,cols,rows,slices);                                        
                     
                     if(h[p]>0) d=d/(2*h[p]*h[p])-1;                     
                     if(d<0) d=0;
              
                     w = exp(-d);   
              
                     fima[p] = fima[p] + w*ima[p1];
                     pesos[p] = pesos[p] + w;
                
                     fima[p1] = fima[p1] + w*ima[p];
                     pesos[p1] = pesos[p1] + w;    
				}		 
            }
			}							
            }                        			
		}
	}
    }

    _endthreadex( 0 );
    return 0;
} 



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

/*Declarations*/
mxArray *xData;
double *ima, *fima,*pesos,*lf,*h;
mxArray *Mxmedias,*Mxpesos,*xtmp;
double *medias,*tmp;
mxArray *pv;
double off,media;
int ini,fin,i,j,k,ii,jj,kk,ni,nj,nk,v,ndim,indice,f,Nthreads,rc,ft;
const int  *dims;
int fac[3];
bool salir;

myargument *ThreadArgs;  
HANDLE *ThreadList; // Handles to the worker threads   

if(nrhs<5) 
{
    printf("Wrong number of arguments!!!\r");
    exit;
}

/*Copy input pointer x*/
xData = prhs[0];

/*Get matrix x*/
ima = mxGetPr(xData);

ndim = mxGetNumberOfDimensions(prhs[0]);
dims= mxGetDimensions(prhs[0]);

pv = prhs[1];
v = (int)(mxGetScalar(pv));
pv = prhs[2];
f = (int)(mxGetScalar(pv));
xData = prhs[3];
h = (double*) mxGetPr(xData);
pv = prhs[4];
lf = (double*)(mxGetPr(pv));
for(i=0;i<3;i++) fac[i]=(int)lf[i];

/*Allocate memory and assign output pointer*/

plhs[0] = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);
Mxmedias = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL); 
xtmp = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);
tmp = mxGetPr(xtmp);

/*Get a pointer to the data space in our newly allocated memory*/
fima = mxGetPr(plhs[0]);
medias = mxGetPr(Mxmedias);

Mxpesos = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL); 
pesos = mxGetPr(Mxpesos);

rc=dims[0]*dims[1];
for(k=0;k<dims[2];k++)
{
for(j=0;j<dims[1];j++)
{
  for(i=0;i<dims[0];i++)
  {
        media=0;
      	for(ii=-1;ii<=1;ii++)
        {
          ni=i+ii;  
          if(ni<0) ni=-ni;
          if(ni>=dims[0]) ni=2*dims[0]-ni-1;            
		  for(jj=-1;jj<=1;jj++)
          {
            nj=j+jj;		   		  
            if(nj<0) nj=-nj;
            if(nj>=dims[1]) nj=2*dims[1]-nj-1;
            for(kk=-1;kk<=1;kk++)
            {              
              nk=k+kk;                                       
              if(nk<0) nk=-nk;
	          if(nk>=dims[2]) nk=2*dims[2]-nk-1;
                           
              media = media + ima[nk*rc+nj*dims[0]+ni];                           
            }
         }
        }
        medias[k*rc+j*dims[0]+i]=media/27;
  }
}
}

ft=fac[2]*fac[1]*fac[0];     
for(k=0;k<dims[2]/fac[2];k++)
for(j=0;j<dims[1]/fac[1];j++)
for(i=0;i<dims[0]/fac[0];i++)
{
    media=0;
    for (kk=0;kk<fac[2];kk++) 
    for (jj=0;jj<fac[1];jj++) 
    for (ii=0;ii<fac[0];ii++) 
        media+=ima[(k*fac[2]+kk)*rc + (j*fac[1]+jj)*dims[0] + i*fac[0]+ii];     
    tmp[k*rc+(j*dims[0])+i]=media/ft;
}

for(k=0;k<dims[2]*dims[1]*dims[0];k++)
{
    fima[k]=ima[k];
    pesos[k]=1.0;
}
//th=0.6*h;  //3/sqrt(27)

Nthreads=dims[2]<8?dims[2]:8;

// Reserve room for handles of threads in ThreadList
ThreadList = (HANDLE*)malloc(Nthreads* sizeof( HANDLE ));
ThreadArgs = (myargument*) malloc( Nthreads*sizeof(myargument));
    
for (i=0; i<Nthreads; i++)
    {       
	// Make Thread Structure
    ini=(i*dims[2])/Nthreads;
    fin=((i+1)*dims[2])/Nthreads;
        
	ThreadArgs[i].cols=dims[0];
    ThreadArgs[i].rows=dims[1];
   	ThreadArgs[i].slices=dims[2];
    ThreadArgs[i].in_image=ima;
	ThreadArgs[i].out_image=fima;
  	ThreadArgs[i].mean_image=medias;
    ThreadArgs[i].pesos=pesos;
    ThreadArgs[i].ini=ini;
    ThreadArgs[i].fin=fin;
    ThreadArgs[i].radio=v;  
    ThreadArgs[i].f=f;      
    //ThreadArgs[i].th=th;    
    ThreadArgs[i].sigma=h;    
	
    ThreadList[i] = (HANDLE)_beginthreadex( NULL, 0, &ThreadFunc, &ThreadArgs[i] , 0, NULL );
  }
    
  for (i=0; i<Nthreads; i++) { WaitForSingleObject(ThreadList[i], INFINITE); }
  for (i=0; i<Nthreads; i++) { CloseHandle( ThreadList[i] ); }
    
  free(ThreadArgs); 
  free(ThreadList);
  

for(i=0;i<dims[0]*dims[1]*dims[2];i++) fima[i]/=pesos[i];

/* apply mean preservation constraint*/

for(k=0;k<dims[2];k=k+fac[2])
for(j=0;j<dims[1];j=j+fac[1])
for(i=0;i<dims[0];i=i+fac[0])
{  
    salir=false;
     
    media=0;
    for (kk=0;kk<fac[2];kk++) 
    for (jj=0;jj<fac[1];jj++) 
    for (ii=0;ii<fac[0];ii++) 
        media+=fima[(k+kk)*rc + (j+jj)*dims[0] + i+ii];     
    media=media/ft;
        
    off=tmp[(k/fac[2])*rc+(j/fac[1])*dims[0]+(i/fac[0])]-media;
        
    for (kk=0;kk<fac[2];kk++) 
    for (jj=0;jj<fac[1];jj++) 
    for (ii=0;ii<fac[0];ii++)
    {
        fima[(k+kk)*rc + (j+jj)*dims[0] + (i+ii)]+=off;                 
    }   
}


return;

}

