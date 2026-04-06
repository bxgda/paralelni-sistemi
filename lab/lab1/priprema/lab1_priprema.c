#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void main(int argc, char *argv[])
{
    int rank, p, i, j;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int broj_radnika = p - 1;
    int velicina_niza = broj_radnika * (broj_radnika + 1);

    if (rank == 0)
    {
        int *niz = (int *)malloc(velicina_niza * sizeof(int));
        for (i = 0; i < velicina_niza; i++)
            niz[i] = i + 1;

        // saljemo svakom radniku 2*i elemenata
        int offset = 0;
        for (i = 1; i <= broj_radnika; i++)
        {
            MPI_Send(niz + offset, 2 * i, MPI_INT, i, 0, MPI_COMM_WORLD);
            offset += 2 * i;
        }

        free(niz);
    }
    else
    {
        // svaki radnik prima 2*rank elemenata
        int broj_elemenata = 2 * rank;
        int *local = (int *)malloc(broj_elemenata * sizeof(int));

        MPI_Recv(local, broj_elemenata, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        int sum = 0;
        for (i = 0; i < broj_elemenata; i++)
            sum += local[i];

        printf("Proces %d: suma = %d\n", rank, sum);

        free(local);
    }

    MPI_Finalize();
}