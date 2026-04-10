/*
    TEKST ZADATKA:

    Napisati MPI program koji realizuje množenje mattrice A N x K i matrice B K x M, 
    čime se dobija i prikazuje rezultujuća matrica C. 
    Takođe program pronalazi i prikazuje minimum elemenata svake vrste matrice A. 
    Izračunavanje se obavlja tako što master proces šalje svakom procesu po K/p kolona matrice A 
    i po K/p vrsta matrice B (p - broj procesa, K deljivo sa p). 
    Master proces inicijalizuje matricu A i matricu B. 
    Predvideti da se slanje svih kolona matrice A jednom procesu obavlja odjenom direktno iz matrice A. 
    Predvideti da se sladnje svih vrsta matrice B jednom procesu obavlja odjednom i direktno iz matrice B. 
    Svi procesi učestvuju u izračunavanjima potrebnim za generisanje rezultata programa. 
    Rezultati programa se nalaze u proizvoljnom procesu koji ih prikazuje. 
    Zadatak realizovati korišćenjem grupnih operacija.
*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <limits.h>

#define N 4
#define K 8
#define M 3

int main(int argc, char *argv[])
{
    // #1
    int A[N][K], B[K][M], C[N][M], minA[N], locC[N * M] = {0}, locMinA[N]; 
    int rank, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // #2
    int po_procesu = K / p; 

    int *locA = (int *)malloc(N * po_procesu * sizeof(int));
    int *locB = (int *)malloc(M * po_procesu * sizeof(int));

    // #3
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++)
                A[i][j] = i + j + 1; 

        for (int i = 0; i < K; i++)
            for (int j = 0; j < M; j++)
                B[i][j] = 1;
    }

    // #4
    MPI_Datatype kolona_tip_v, kolona_tip;
    MPI_Type_vector(N, po_procesu, K, MPI_INT, &kolona_tip_v);
    MPI_Type_create_resized(kolona_tip_v, 0, po_procesu * sizeof(int), &kolona_tip);
    MPI_Type_commit(&kolona_tip);

    MPI_Scatter(A, 1, kolona_tip, locA, N * po_procesu, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, M * po_procesu, MPI_INT, locB, M * po_procesu, MPI_INT, 0, MPI_COMM_WORLD);

    // #5
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < po_procesu; k++)
                locC[i * M + j] += locA[i * po_procesu + k] * locB[k * M + j];

    for (int i = 0; i < N; i++)
    {
        locMinA[i] = INT_MAX;
        for (int j = 0; j < po_procesu; j++)
        {
            if (locA[i * po_procesu + j] < locMinA[i])
                locMinA[i] = locA[i * po_procesu + j];
        }
    }

    // #6
    MPI_Reduce(locC, C, N * M, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(locMinA, minA, N, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // #7
    if (rank == 0)
    {
        printf("Matrica C (A*B):\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
                printf("%d ", C[i][j]);
            printf("\n");
        }

        printf("\nMinimum svake vrste matrice A:\n");
        for (int i = 0; i < N; i++)
            printf("minA[%d] = %d\n", i, minA[i]);
    }

    // #8
    free(locA);
    free(locB);
    
    MPI_Type_free(&kolona_tip_v);
    MPI_Type_free(&kolona_tip);

    MPI_Finalize();
    return 0;
}

/*
    #1 - mpi inicijalizacija
    #2 - racunanje koliko ce svaki proces da dobije kolona/vrsta i dinamicka alokacija tamo gde dimenzija zavisi od broja procesa
    #3 - proces 0 inicijalizuje matrice
    #4 - pravljenje posebnog tipa za kolone i rasturanje matrice A po kolonama, a matrice B po vrstama
    #5 - mnozenje (potpuno isto kao ranije) i trazenje lokalnih minimuma za A, svako nalazi svoj minimum a na kraju se ti minimumi uporedjuju
    #6 - REDUKCIJE: MPI_SUM za matricu C, i MPI_MIN da bi od lokalnih minimuma dobili prave globalne minimume za sve vrste
    #7 - stampanje rezultata programa
    #8 - dealokacija memorije / sprecavanje memory leak-a samo za malloc-ovane nizove
*/
