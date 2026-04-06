/*
    TEKST ZADATKA:

    Napisati MPI program koji realizuje množenje matrice A(n×k) i matrice B(k×m),
    čime se dobija rezultujuća matrica C. Množenje se obavlja tako što master proces šalje svakom procesu celu matricu A
    i po m/p kolona matrice B (p-broj procesa, m deljivo sa p).
    Elementi svih kolona matrice B koji se šalju svakom procesu, šalju se odjednom.
    Svi procesi učestvuju u izračunavanjima potrebnim za generisanje rezultata programa.
    Konačni rezultat množenja se nalazi u procesu koji ga i prikazuje, a koji sadrži maksimalnu vrednost elemenata u matrici B,
    nakon raspodele kolona po procesima. Zadatak realizovati korišćenjem grupnih operacija.
    Dati primer matrica A i B za proizvoljno n, k i m i ilustrovati podelu podataka po procesima, kao i način generisanja rezultujuće matrice C.
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

    // #3
    MPI_Reduce(&localMax, &globalMax, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalMax, 1, MPI_2INT, 0, MPI_COMM_WORLD);

    // #4
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
    #1 svako ucestvuje u potpunom izracunavanju "kolona_po_procesu" kolona matrice C
    #2 pravimo strukturu kao na vezbama kako bi nasli maksimalni element i njegov rank... prvo nalazi svako svoj maksimum
    #3 za redukcijom MPI_MAXLOC nalazimo makksimum od svih maksimalnih elemenata iz strukture i upisuje se u global taj maksimum i rank iz te strukture koja je imala maksimum maksimuma
    #4 posto svaki proces ima izracunate kolone, da bi to sakupili treba nam tip za tu kolonu da bi se to u matrici C sakupilo bas kao kolona
*/