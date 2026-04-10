/*
    TEKST ZADATKA:

    Napisati MPI program koji realizuje množenje matrice Aₙₓₖ i matrice Bₖₓₘ,
    čime se dobija i prikazuje rezultujuća matrica C.
    Takođe, program pronalazi i prikazuje minimum elemenata svake vrste matrice B
    (p- broj procesa, k deljivo sa p). Master proces inicijalizuje matricu A i matricu B.
    Predvideti da se slanje svih kolona matrice A jednom procesu obavlja odjednom i direktno iz matrice A.
    Predvideti da se slanje svih vrsta matrice B jednom procesu obavlja odjednom i direktno iz matrice B.
    Svi procesi učestvuju u izračunavanjima potrebnim za generisanje rezultata programa.
    Rezultati programa se nalaze u proizvoljnom procesu koji ih i prikazuje.
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
    int A[N][K], B[K][M], C[N][M], minB[K];
    int rank, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // #2
    int po_procesu = K / p; // kolona po procesu = vrsta po procesu

    int *locA = (int *)malloc(N * po_procesu * sizeof(int));
    int *locB = (int *)malloc(M * po_procesu * sizeof(int));
    int *locC = (int *)calloc(N * M, sizeof(int));
    int *locMinB = (int *)malloc(po_procesu * sizeof(int));

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

    for (int i = 0; i < po_procesu; i++)
    {
        locMinB[i] = INT_MAX;
        for (int j = 0; j < M; j++)
            if (locB[i * M + j] < locMinB[i])
                locMinB[i] = locB[i * M + j];
    }

    // #6
    MPI_Reduce(locC, C, N * M, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(locMinB, po_procesu, MPI_INT, minB, po_procesu, MPI_INT, 0, MPI_COMM_WORLD);

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

        printf("\nMinimum svake vrste matrice B:\n");
        for (int i = 0; i < K; i++)
            printf("minB[%d] = %d\n", i, minB[i]);
    }

    // #8
    free(locA);
    free(locB);
    free(locC);
    free(locMinB);

    MPI_Finalize();
}

/*
    #1 - alociranje memorije za stvari koje vec znamo koliko zauzimaju i mpi inicijalizacija
    #2 - racunanje koliko ce svaki proces da dobije kolona i vrsta i dinamicka alokacija potrebnih lokalnih promenljivih
    #3 - proces 0 inicijalizuje matrice
    #4 - pravljenje posebnog tipa za kolone i distribucija kolona i vrsta
    #5 - izracunavanje rezultujuce matrice (svaki proces ucestvuje u delu izracunavanja cele rezultujuce matrice)
         izracunavanje minimuma (svaki proces izracunava minimum za vrste koje ima)
    #6 - sakupljanje rezultata reduce za matricu i gather za minimum
    #7 - stampanje rezultata
    #8 - nikako mem-leak
*/
