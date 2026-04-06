/*
    TEKST ZADAKA:

    Napisati MPI program koji realizuje množenje matrice A(n×k) i matrice B(k×m),
    čime se dobija rezultujuća matrica C. Množenje se obavlja tako što master proces šalje svakom procesu celu matricu A
    i po m/p kolona matrice B (p-broj procesa, m deljivo sa p).
    Elementi svih kolona matrice B koji se šalju svakom procesu, šalju se odjednom.
    Svi procesi učestvuju u izračunavanjima potrebnim za generisanje rezultata programa.
    Zadatak realizovati korišćenjem grupnih operacija.
*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 4
#define K 6
#define M 8

int main(int argc, char **argv)
{
    int A[N][K], B[K][M], C[N][M];
    int rank, p;

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

    MPI_Datatype tip_kolona_v, tip_kolona;
    MPI_Type_vector(K, kolona_po_procesu, M, MPI_INT, &tip_kolona_v);
    MPI_Type_create_resized(tip_kolona_v, 0, kolona_po_procesu * sizeof(int), &tip_kolona);
    MPI_Type_commit(&tip_kolona);

    MPI_Scatter(B, 1, tip_kolona, locB, K * kolona_po_procesu, MPI_INT, 0, MPI_COMM_WORLD);

    // #1
    for (int i = 0; i < N; i++)
        for (int j = 0; j < kolona_po_procesu; j++)
            for (int k = 0; k < K; k++)
                locC[i * kolona_po_procesu + j] += A[i][k] * locB[k * kolona_po_procesu + j];

    // #2
    MPI_Datatype kolona_C_v, kolona_C;
    MPI_Type_vector(N, kolona_po_procesu, M, MPI_INT, &kolona_C_v);
    MPI_Type_create_resized(kolona_C_v, 0, kolona_po_procesu * sizeof(int), &kolona_C);
    MPI_Type_commit(&kolona_C);

    MPI_Gather(locC, N * kolona_po_procesu, MPI_INT, C, 1, kolona_C, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
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
    #1 svaki proces ucestvuje potpuno u izracunavanju po "kolona_po_procesu" kolona matrice C
    #2 posto u locC imamo samo obican niz a mi treba da ga gledamo kao kolone, moramo da napravimo tip i za te kolone da kada uradimo gather
       da se u matrici C to spakuje po kolonama a ne kao niz po niz (ispalo bi da se te izracunate kolone spakovale kao vrste da nismo napravili izvedeni tip)
*/