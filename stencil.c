/* Copyright (c) 2015 The University of Edinburgh. */

/* 
* This software was developed as part of the                       
* EC FP7 funded project Adept (Project ID: 610490)                 
* www.adept-project.eu                                            
*/

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

/*
 * MPI Stencil benchmark
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

#include "level1.h"
#include "utils.h"

#define REPS 100

void double_stencil27(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/26;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void float_stencil27(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/26;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void int_stencil27(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/26; // Leave as float otherwise it ends up round to 0 which causes everything to be 0

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (int) rand()/ (int)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}




void double_stencil19(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void float_stencil19(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil19(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (int) rand()/ (int)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Recv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Recv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void double_stencil9(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void float_stencil9(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil9(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (int) rand()/ (int)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void double_stencil5(unsigned int size){


  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }

  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);

      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 5 point");
  }



  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void float_stencil5(unsigned int size){


  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  float fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }

  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);

      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 5 point");
  }



  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil5(unsigned int size){


  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }

  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (int) rand()/ (int)(1.0 + RAND_MAX);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req;

  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req);
      /* Recv from lower PE */
      MPI_Recv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      /* Recv from higher PE */
      MPI_Recv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req);
      MPI_Recv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);

      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]

                          ) * fac;
        }
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 5 point");
  }



  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void double_stencil27_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/26;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req4);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }

    

    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }

    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }



    MPI_Barrier(MPI_COMM_WORLD);


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void float_stencil27_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/26;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req4);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }

    

    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }


    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void int_stencil27_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/26; // Leave as float otherwise it ends up round to 0 which causes everything to be 0

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("27-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start, end;

  /* zero all of array (including halos) */

  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }


  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (int) rand()/ (int)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {
    /* Before we start, do a bit of HALO swapping */
    // SEND from local_size-2*size*size, a layer of size*size to N+1
    // RECV into 0, a layer of size*size from N+1


    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req4);

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }

    


    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
                                      a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
                                      a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }

    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
					a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
					a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
					a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
					a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


    MPI_Barrier(MPI_COMM_WORLD);


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 27 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}




void double_stencil19_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req4);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)a0, (size*size), MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }

    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }


    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }


    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}

void float_stencil19_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  float fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)a0, (size*size), MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }

    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil19_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, k, iter;
  int n = size-2;
  double fac = 1.0/18;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*size*(local_size));

  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("19-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i*size*size+j*size+k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      for (k = 1; k < n+1; k++) {
        a0[i*size*size+j*size+k] = (int) rand()/ (int)(1.0 + RAND_MAX);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {

    /* printf("%d before swaps\n", world_rank); */

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 2nd recv\n", world_rank); */

      /* Send to lower PE */
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* printf("%d after 2nd send\n", world_rank); */
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size*size)), (size*size), MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
      /* printf("%d after 1st recv\n", world_rank); */

    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+(size*size)), (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      /* printf("%d after 1st send\n", world_rank); */
      MPI_Irecv((void*)a0, (size*size), MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
      /* printf("%d after 1st recv\n", world_rank); */
    }


    for (i = 2; i < local_n; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a1[i*size*size+j*size+k] = (
                                      a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
                                      a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
                                      a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
                                      a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

                                      a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
                                      a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

                                      a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
                                      a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

                                      a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
                                      ) * fac;
        }
      }
    }

    /* wait for incoming comms to complete */
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if ((world_rank > 0) && (world_rank < (world_size-1))) {
	MPI_Wait(&req4, MPI_STATUS_IGNORE);
    }
      
    /* update edge points*/
    i = 1;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }
    i = local_n;
    for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
	    a1[i*size*size+j*size+k] = (
					a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
					a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
					a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
					a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

					a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
					a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

					a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
					a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

					a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
					) * fac;
        }
    }

    for (i = 1; i < local_n+1; i++) {
      for (j = 1; j < n+1; j++) {
        for (k = 1; k < n+1; k++) {
          a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */

  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 19 point");
  }
  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void double_stencil9_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req4);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }


      for (i = 2; i < local_n; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }

      /* wait for incoming comms to complete */
      MPI_Wait(&req2, MPI_STATUS_IGNORE);
      if ((world_rank > 0) && (world_rank < (world_size-1))) {
	  MPI_Wait(&req4, MPI_STATUS_IGNORE);
      }
      
      /* update edge points*/
      i = 1;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
      }
      i = local_n;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
      }


      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


    MPI_Barrier(MPI_COMM_WORLD);



  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void float_stencil9_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }


      for (i = 2; i < local_n; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }

      /* wait for incoming comms to complete */
      MPI_Wait(&req2, MPI_STATUS_IGNORE);
      if ((world_rank > 0) && (world_rank < (world_size-1))) {
	  MPI_Wait(&req4, MPI_STATUS_IGNORE);
      }
      
      /* update edge points*/
      i = 1;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
      }
      i = local_n;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
      }



      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }

    MPI_Barrier(MPI_COMM_WORLD);


  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil9_overlapped(unsigned int size){

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* printf("Local n on Rank %d is %d, total n is %d\n", world_rank, local_n, n); */



  /* Work buffers, with halos */
  int *a0 = (int*)malloc(sizeof(int)*size*(local_size));
  int *a1 = (int*)malloc(sizeof(int)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }


  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (int) rand()/ (int)(1.0 + RAND_MAX);
    }
  }

  /* run main computation on host */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  /* run main computation on host */
  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }


      for (i = 2; i < local_n; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

                          ) * fac;
        }
      }

      /* wait for incoming comms to complete */
      MPI_Wait(&req2, MPI_STATUS_IGNORE);
      if ((world_rank > 0) && (world_rank < (world_size-1))) {
	  MPI_Wait(&req4, MPI_STATUS_IGNORE);
      }

      /* update edge points */
      i = 1;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]
			  
                          ) * fac;
      }
      i = local_n;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j] +
                          a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
                          a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]
			  
                          ) * fac;
      }
      
      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 9 point");
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void double_stencil5_overlapped(unsigned int size){


  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  double fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  double *a0 = (double*)malloc(sizeof(double)*size*(local_size));
  double *a1 = (double*)malloc(sizeof(double)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }

  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, &req4);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, size, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }


      for (i = 2; i < local_n; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]

                          ) * fac;
        }
      }

      /* wait for incoming comms to complete */
      MPI_Wait(&req2, MPI_STATUS_IGNORE);
      if ((world_rank > 0) && (world_rank < (world_size-1))) {
	  MPI_Wait(&req4, MPI_STATUS_IGNORE);
      }
      
      /* update edge points*/
      i = 1;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]
			  
                          ) * fac;
      }
      i = local_n;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]
			  
                          ) * fac;
      }

      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }

    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 5 point");
  }



  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void float_stencil5_overlapped(unsigned int size){


  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

  int i, j, iter;
  int n = size-2;
  float fac = 1.0/8;

  /*
   * Compute size of block each rank will work on
   * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
   * in the MPI case as in the serial case.
   */

  int local_n = 0;
  int local_size = 0;
  if (world_rank != 0){
    local_n = n / world_size;
  }
  else if (world_rank == 0){
    local_n = (n/world_size) + (n%world_size);
  }
  else{
    printf("Some error occured in size calculation\n");
  }

  local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

  /* Work buffers, with halos */
  float *a0 = (float*)malloc(sizeof(float)*size*(local_size));
  float *a1 = (float*)malloc(sizeof(float)*size*(local_size));


  if(a0==NULL||a1==NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf("9-point Stencil Error: Unable to allocate memory\n");
  }

  struct timespec start,end;
  /* zero all of array (including halos) */
  for (i = 0; i < local_size; i++) {
    for (j = 0; j < size; j++) {
      a0[i*size+j] = 0.0;
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < local_n+1; i++) {
    for (j = 1; j < n+1; j++) {
      a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Only do the timings on the one rank */
  if (world_rank == 0){
    clock_gettime(CLOCK, &start);
  }
  MPI_Request req1, req2, req3, req4;

  for (iter = 0; iter < REPS; iter++) {

    if (world_rank == 0 && world_size>1){
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
    }
    if (world_rank>0 && world_rank<(world_size-1)){
      /* Send to higher PE */
      MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
      /* Recv from lower PE */
      MPI_Irecv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

      /* Send to lower PE */
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
      /* Recv from higher PE */
      MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
    }
    if (world_rank == (world_size-1) && world_size>1){
      MPI_Isend((void*)(a0+size), size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
      MPI_Irecv((void*)a0, size, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
    }


      for (i = 2; i < local_n; i++) {
        for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]

                          ) * fac;
        }
      }

      /* wait for incoming comms to complete */
      MPI_Wait(&req2, MPI_STATUS_IGNORE);
      if ((world_rank > 0) && (world_rank < (world_size-1))) {
	  MPI_Wait(&req4, MPI_STATUS_IGNORE);
      }
      
      /* update edge points*/
      i = 1;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]
			  
                          ) * fac;
      }
      i = local_n;
      for (j = 1; j < n+1; j++) {
          a1[i*size+j] = (
                          a0[i*size+(j-1)] + a0[i*size+(j+1)] +
                          a0[(i-1)*size+j] + a0[(i+1)*size+j]
			  
                          ) * fac;
      }
      

      for (i = 1; i < local_n+1; i++) {
        for (j = 1; j < n+1; j++) {
          a0[i*size+j] = a1[i*size+j];
        }
      }


    MPI_Barrier(MPI_COMM_WORLD);

  } /* end iteration loop */
  if (world_rank == 0){
    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Stencil - 5 point");
  }



  /* Free malloc'd memory to prevent leaks */
  free(a0);
  free(a1);
}


void int_stencil5_overlapped(unsigned int size){


    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    int i, j, iter;
    int n = size-2;
    double fac = 1.0/8;

    /*
     * Compute size of block each rank will work on
     * We do this in a slightly odd method to try an ensure we have the same number of data plane layers (n)
     * in the MPI case as in the serial case.
     */

    int local_n = 0;
    int local_size = 0;
    if (world_rank != 0){
	local_n = n / world_size;
    }
    else if (world_rank == 0){
	local_n = (n/world_size) + (n%world_size);
    }
    else{
	printf("Some error occured in size calculation\n");
    }

    local_size = local_n + 2; // Each PE needs to have a halo on both edges in 3D. n is size of the data.

    /* Work buffers, with halos */
    int *a0 = (int*)malloc(sizeof(int)*size*(local_size));
    int *a1 = (int*)malloc(sizeof(int)*size*(local_size));


    if(a0==NULL||a1==NULL){
	/* Something went wrong in the memory allocation here, fail gracefully */
	printf("9-point Stencil Error: Unable to allocate memory\n");
    }

    struct timespec start,end;
    /* zero all of array (including halos) */
    for (i = 0; i < local_size; i++) {
	for (j = 0; j < size; j++) {
	    a0[i*size+j] = 0.0;
	}
    }

    /* use random numbers to fill interior */
    for (i = 1; i < local_n+1; i++) {
	for (j = 1; j < n+1; j++) {
	    a0[i*size+j] = (int) rand()/ (int)(1.0 + RAND_MAX);
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Only do the timings on the one rank */
    if (world_rank == 0){
	clock_gettime(CLOCK, &start);
    }
    MPI_Request req1, req2, req3, req4;

    for (iter = 0; iter < REPS; iter++) {
	
	/* start asynchronous halo communications */
	if (world_rank == 0 && world_size>1){
	    /* Send to higher PE */
	    MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
	    /* Recv from higher PE */
	    MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req2);
	}
	if (world_rank>0 && world_rank<(world_size-1)){
	    /* Send to higher PE */
	    MPI_Isend((void*)(a0+((local_size-2)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req1);
	    /* Recv from lower PE */
	    MPI_Irecv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);

	    /* Send to lower PE */
	    MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req3);
	    /* Recv from higher PE */
	    MPI_Irecv((void*)(a0+((local_size-1)*size)), size, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, &req4);
	}
	if (world_rank == (world_size-1) && world_size>1){
	    /* Send to lower PE */
	    MPI_Isend((void*)(a0+size), size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req1);
	    /* Receive from lower PE */
	    MPI_Irecv((void*)a0, size, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &req2);
	}


	/*MPI_Barrier(MPI_COMM_WORLD);*/
	/* update inner points */
	for (i = 2; i < local_n; i++) {
	    for (j = 1; j < n+1; j++) {
		a1[i*size+j] = (
				a0[i*size+(j-1)] + a0[i*size+(j+1)] +
				a0[(i-1)*size+j] + a0[(i+1)*size+j]

				) * fac;
	    }
	}

	/* wait for incoming comms to complete */
	MPI_Wait(&req2, MPI_STATUS_IGNORE);
	if ((world_rank > 0) && (world_rank < (world_size-1))) {
	    MPI_Wait(&req4, MPI_STATUS_IGNORE);
	}

	/* update edge points */
	i = 1;
	for (j = 1; j < n+1; j++) {
	    a1[i*size+j] = (
			    a0[i*size+(j-1)] + a0[i*size+(j+1)] +
			    a0[(i-1)*size+j] + a0[(i+1)*size+j]
			    
			    ) * fac;
	}
	i = local_n;
	for (j = 1; j < n+1; j++) {
	    a1[i*size+j] = (
			    a0[i*size+(j-1)] + a0[i*size+(j+1)] +
			    a0[(i-1)*size+j] + a0[(i+1)*size+j]
			    
			    ) * fac;
	}

	/* copy to other buffer */
	for (i = 1; i < local_n+1; i++) {
	    for (j = 1; j < n+1; j++) {
		a0[i*size+j] = a1[i*size+j];
	    }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
    } /* end iteration loop */
    if (world_rank == 0){
	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Stencil - 5 point");
    }



    /* Free malloc'd memory to prevent leaks */
    free(a0);
    free(a1);
}
