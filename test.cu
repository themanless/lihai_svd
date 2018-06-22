#include"svd.cuh"
#include"time.cuh"
#include<stdio.h>
#include<stdlib.h>
int main(int argc,char* argv[]){
	int m;
	int n;
	int batch;
	printf("input m n batch:\n");
	scanf("%d%d%d",&m,&n,&batch);
	double time1;
	double time2;
	time1=gettime();
	cuComplex* T=(cuComplex*)malloc(sizeof(cuComplex)*m*n*batch);
	cuComplex* u=(cuComplex*)malloc(sizeof(cuComplex)*m*((m<n)?m:n));
	cuComplex* v=(cuComplex*)malloc(sizeof(cuComplex)*n*((m<n)?m:n));
	float* s=(float*)malloc(sizeof(float)*((m<n)?m:n));
	cuComplex* U=(cuComplex*)malloc(sizeof(cuComplex)*batch*m*((m<n)?m:n));
	cuComplex* V=(cuComplex*)malloc(sizeof(cuComplex)*batch*n*((m<n)?m:n));
	float* S=(float*)malloc(sizeof(float)*batch*((m<n)?m:n));
	srand((unsigned)time(NULL));
	for(int i=0;i<m*n*batch;i++){
       T[i].x=(float)rand()/(RAND_MAX/100);
       T[i].y=(float)rand()/(RAND_MAX/100);
	}
	cuComplex** temp=(cuComplex**)T;
	cuComplex (*tmp)[m*n]=(cuComplex (*)[m*n])temp;
	cuComplex* swap=(cuComplex*)malloc(sizeof(cuComplex)*m*n);
	for(int i=0;i<batch;i++){
		for(int j=0;j<n*m;j++){
         swap[j]=tmp[i][j];
		}

	  svd(m,n,swap,u,v,s);
		    for(int j=0;j<m*((m<n)?m:n);j++){

		    	 U[j+i*m*((m<n)?m:n)].x=u[j].x;
		    	 U[j+i*m*((m<n)?m:n)].y=u[j].y;

		     }

		    for(int j=0;j<n*((m<n)?m:n);j++){

		        V[j+i*n*((m<n)?m:n)].x=v[j].x;
		        V[j+i*n*((m<n)?m:n)].y=v[j].y;
		    }

		   for(int j=0;j<((m<n)?m:n);j++){
		    	S[j+i*((m<n)?m:n)]=s[j];
		    }
	}
	time2=gettime();
	printf("the time of svd :%f s\n",time2-time1);
return 1;
}
