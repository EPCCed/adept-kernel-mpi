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

void bench_level1(char *, unsigned int, unsigned int, char *, char *);

int int_dot_product(unsigned int);
int float_dot_product(unsigned int);
int double_dot_product(unsigned int);

int int_scalar_mult(unsigned int);
int float_scalar_mult(unsigned int);
int double_scalar_mult(unsigned int);

int double_norm(unsigned int);
int float_norm(unsigned int);
int int_norm(unsigned int);

int int_axpy(unsigned int);
int float_axpy(unsigned int);
int double_axpy(unsigned int);

int int_dmatvec_product(unsigned int);
int float_dmatvec_product(unsigned int);
int double_dmatvec_product(unsigned int);

int double_spmatvec_product(unsigned long);
int float_spmatvec_product(unsigned long);

void double_stencil27(unsigned int);
void float_stencil27(unsigned int);
void int_stencil27(unsigned int);

void double_stencil19(unsigned int);
void float_stencil19(unsigned int);
void int_stencil19(unsigned int);

void double_stencil9(unsigned int);
void float_stencil9(unsigned int);
void int_stencil9(unsigned int);

void double_stencil5(unsigned int);
void float_stencil5(unsigned int);
void int_stencil5(unsigned int);

void fileparse(unsigned int);
