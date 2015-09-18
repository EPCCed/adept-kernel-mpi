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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <ctype.h>

#include "utils.h"
#include "level1.h"

int create_line(char*, size_t, char*, unsigned int);
int seek_match(char*, size_t, char*, unsigned int);

void fileparse(unsigned int num_rows) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    char search_phrase[] = "AdeptProject";
    size_t sp_len = strlen(search_phrase);

    unsigned int desired_line_len = 81;
    char line[desired_line_len];

    srand(time(NULL)); // Set seed

    int i = 0;
    int r = 0;
    int m = 0;
    int mismatch = 0;
    int r_count = 0;
    int m_count = 0;
    struct timespec start, end;

    /* p_num_rows is the number of rows across all processes */
    unsigned int p_num_rows;
    p_num_rows = (unsigned int) (num_rows * world_size);


    /* Generate (on the fly) the test file for the run */
    /* Make this single threaded for ease */
    if (world_rank == 0) {
        FILE* fp;
        fp = fopen("testfile", "w+");

        for (i = 0; i < p_num_rows; i++) {
            r = create_line(search_phrase, sp_len, line, desired_line_len);
            m = seek_match(search_phrase, sp_len, line, desired_line_len);
            if (r != m) {
                mismatch++;
            }
            if (r == 0) {
                r_count++;
            }
            if (m == 0) {
                m_count++;
            }
            fprintf(fp, "%s\n", line);
        }
        fsync(fileno(fp));
        fclose(fp);

    }

    m = 0;


    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File fh;
    MPI_Status status;

    /* For holding the data from the file before parsing */
    char *lb = (char*) malloc(sizeof (char)*num_rows * (desired_line_len + 1));
    char *lbp = NULL;


    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        clock_gettime(CLOCK, &start);
    }
    m_count = 0;

    /* This part should use MPI-IO */
    MPI_File_open(MPI_COMM_WORLD, "testfile", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
    MPI_File_read_at(fh, world_rank * num_rows * (desired_line_len + 1), lb, num_rows * (desired_line_len + 1), MPI_CHAR, &status);
    for (i = 0; i < num_rows; i++) {
        lbp = &lb[i * (desired_line_len + 1)];
        m = seek_match(search_phrase, sp_len, lbp, desired_line_len);
        if (m == 0) {
            m_count++;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);

    if (world_rank == 0) {
        clock_gettime(CLOCK, &end);
        elapsed_time_hr(start, end, "Fileparse");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    unlink("testfile"); // Use this to ensure the generated file is removed from the system upon finish

}

/*
 * Create a line of random characters
 * Line will be ll long and appears in l
 * Randomly, phrase contained in sp and of sp_len length will be added to l at a random position
 */
int create_line(char* sp, size_t sp_len, char* l, unsigned int ll) {


    int i = 0;
    int r = 0;
    int flag = 0;

    for (i = 0; i < ll; i++) {
        r = (rand() % 128);
        while (!isalnum(r)) {
            r = (rand() % 128);
        }
        l[i] = (char) r;
    }
    l[i + 1] = '\0';

    r = rand() % 2;

    if (r == 0) {
        flag = 0;
        r = rand() % (ll - sp_len);
        for (i = 0; i < sp_len; i++) {
            l[r + i] = sp[i];
        }
    } else {
        flag = 1;
    }

    return flag;
}

/*
 * Naive matching algorithm
 */
int seek_match(char* sp, size_t sp_len, char* l, unsigned int ll) {

    int i = 0;
    int flag = 1;
    for (i = 0; i < ll - sp_len; i++) {
        if (l[i] == sp[0]) {
            if (strncmp(&l[i], &sp[0], sp_len) == 0) {
                flag = 0;
                break;
            }
        }
    }

    return flag;
}
