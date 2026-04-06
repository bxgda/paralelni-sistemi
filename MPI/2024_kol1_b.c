/*
    TEKST ZADATKA:

    ...2024 kol1 a)
    b) Distribuciju kolona matrice B realizovati korišćenjem Point-to-Point operacija.
*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <limits.h>

#define N 4
#define K 6
#define M 8

int main(int argc, char **argv)
{
    int A[N][K], B[K][M], C[N][M];
    int rank, p;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int kolona_po_procesu = M / p;

    int *locB = (int *)malloc(K * kolona_po_procesu * sizeof(int));
    int *locC = (int *)calloc(N * kolona_po_procesu, sizeof(int));

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++)
                A[i][j] = i + j + 1;

        for (int i = 0; i < K; i++)
            for (int j = 0; j < M; j++)
                B[i][j] = i + j + 1;
    }

    MPI_Bcast(A, N * K, MPI_INT, 0, MPI_COMM_WORLD);

    // P2P distribucija kolona matrice B
    MPI_Datatype tip_kolona_v, tip_kolona;
    MPI_Type_vector(K, kolona_po_procesu, M, MPI_INT, &tip_kolona_v);
    MPI_Type_create_resized(tip_kolona_v, 0, kolona_po_procesu * sizeof(int), &tip_kolona);
    MPI_Type_commit(&tip_kolona);

    // #1
    if (rank == 0)
    {
        // P0 kopira svoje kolone direktno
        for (int i = 0; i < K; i++)
            for (int j = 0; j < kolona_po_procesu; j++)
                locB[i * kolona_po_procesu + j] = B[i][j];

        // P0 salje ostalim procesima
        for (int proces = 1; proces < p; proces++)
            MPI_Send(&B[0][proces * kolona_po_procesu], 1, tip_kolona, proces, 0, MPI_COMM_WORLD);
    }
    else
        MPI_Recv(locB, K * kolona_po_procesu, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < kolona_po_procesu; j++)
            for (int k = 0; k < K; k++)
                locC[i * kolona_po_procesu + j] += A[i][k] * locB[k * kolona_po_procesu + j];

    struct
    {
        int val;
        int rank;
    } localMax, globalMax;

    localMax.val = INT_MIN;
    localMax.rank = rank;

    for (int i = 0; i < K * kolona_po_procesu; i++)
        if (locB[i] > localMax.val)
            localMax.val = locB[i];

    MPI_Reduce(&localMax, &globalMax, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalMax, 1, MPI_2INT, 0, MPI_COMM_WORLD);

    MPI_Datatype kolona_C_v, kolona_C;
    MPI_Type_vector(N, kolona_po_procesu, M, MPI_INT, &kolona_C_v);
    MPI_Type_create_resized(kolona_C_v, 0, kolona_po_procesu * sizeof(int), &kolona_C);
    MPI_Type_commit(&kolona_C);

    MPI_Gather(locC, N * kolona_po_procesu, MPI_INT, C, 1, kolona_C, globalMax.rank, MPI_COMM_WORLD);

    if (rank == globalMax.rank)
    {
        printf("Rezultat prikazuje proces %d (maksimum B = %d)\n\n",
               globalMax.rank, globalMax.val);
        printf("Matrica C (A*B):\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
                printf("%d ", C[i][j]);
            printf("\n");
        }
    }

    free(locB);
    free(locC);

    MPI_Finalize();
}

/*
    #1 P0 sam sebi prvo salje odgovarajuce kolone i onda svim ostalim procesima kolone koje oni treba da dobiju
    ... ostale napomene su kao u 2024 kol1 a)
*/