/*
 * svd.cuh
 *
 *  Created on: Jun 19, 2018
 *      Author: haili
 */

#ifndef SVD_CUH_
#define SVD_CUH_
#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<assert.h>
#include<cuda_runtime.h>
#include"cusolverDn.h"
 void svd(int,int,cuComplex*,cuComplex*,cuComplex*,float*);

#endif /* SVD_CUH_ */
