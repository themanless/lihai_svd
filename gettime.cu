#include "time.cuh"
double gettime(){
	double tseconds=0.0;
	struct timeval mytime;
	gettimeofday(&mytime,(struct timezone*)0);
	tseconds=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
	return tseconds;
}
