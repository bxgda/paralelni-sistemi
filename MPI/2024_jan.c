#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3
#define K 15

void main(int argc, char **argv)
{
    int A[K][N], B[N], C[K];
    int rank, p;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int svoji_redovi = (int)pow(2, rank);

    if (rank == 0)
    {
        for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = i + j + 1;

        for (int i = 0; i < N; i++)
            B[i] = 1;
    }

    MPI_Bcast(B, N, MPI_INT, 0, MPI_COMM_WORLD);

    int *locA = (int *)malloc(svoji_redovi * N * sizeof(int));
    int *locC = (int *)calloc(svoji_redovi, sizeof(int));

    if (rank == 0)
    {
        for (int i = 0; i < svoji_redovi; i++)
            for (int j = 0; j < N; j++)
                locA[i * N + j] = A[i][j];

        int pomeraj = svoji_redovi;
        for (int i = 1; i < p; i++)
        {
            int redovi_za_poslati = (int)pow(2, i);
            MPI_Send(&A[pomeraj][0], redovi_za_poslati * N, MPI_INT, i, 0, MPI_COMM_WORLD);
            pomeraj += redovi_za_poslati;
        }
    }
    else
        MPI_Recv(locA, svoji_redovi * N, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    for (int i = 0; i < svoji_redovi; i++)
        for (int j = 0; j < N; j++)
            locC[i] += locA[i * N + j] * B[j];

    if (rank == p - 1)
    {
        int pomeraj = K - svoji_redovi;
        for (int i = 0; i < svoji_redovi; i++)
            C[pomeraj + i] = locC[i];

        pomeraj = 0;
        for (int i = 0; i < p - 1; i++)
        {
            int redovi_za_primiti = (int)pow(2, i);
            MPI_Recv(&C[pomeraj], redovi_za_primiti, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            pomeraj += redovi_za_primiti;
        }
    }
    else
        MPI_Send(locC, svoji_redovi, MPI_INT, p - 1, 0, MPI_COMM_WORLD);

    if (rank == p - 1)
    {
        printf("Rezultujući vektor C:\n");
        for (int i = 0; i < K; i++)
            printf("%d ", C[i]);
        printf("\n");
    }

    free(locA);
    free(locC);

    MPI_Finalize();
}