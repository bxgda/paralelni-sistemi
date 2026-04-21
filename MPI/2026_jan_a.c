/*

   Napisati MPI program koji realizuje množenje matrice Aₙₓₛ i matrice Bₛₓₖ, čime se dobija rezultujuća matrica C.
   Program pronalazi i prikazuje i proizvod svake vrste matrice B. Matrice A i B se inicijalizuju u master procesu.
   Master proces šalje svakom procesu celu matricu A. Matrica B je podeljena u blokove po kolonama (k je deljivo sa p) i
   to tako da proces Pᵢ dobija kolone sa indeksima l, l mod p = i, tj. kolone sa indeksima i, i+p, i+2p, ..., i+k-p.
   Predvideti da se slanje kolona matrice B svakom procesu obavlja odjednom i direktno iz matrice B. Svaki proces
   obavlja odgovarajuća izračunavanja i učestvuje u generisanju rezultata. Rezultate programa prikazati u proizvoljnom
   procesu. Zadatak rešiti korišćenjem grupnih operacija.

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2
#define S 3
#define K 8

void main(int argc, char **argv)
{
    int A[N][S], B[S][K], C[N][K], prodB[S], locProdB[S];
    int rank, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int kolona_po_procesu = K / p;

    int *locB = (int *)malloc(S * kolona_po_procesu * sizeof(int));
    int *locC = (int *)calloc(N * kolona_po_procesu, sizeof(int));

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < S; j++)
                A[i][j] = i + j + 1;

        for (int i = 0; i < S; i++)
            for (int j = 0; j < K; j++)
                B[i][j] = i + j + 1;
    }

    MPI_Bcast(A, N * S, MPI_INT, 0, MPI_COMM_WORLD);

    // #1
    MPI_Datatype B_kolona_v, B_kolona;
    MPI_Type_vector(S * kolona_po_procesu, 1, p, MPI_INT, &B_kolona_v);
    MPI_Type_create_resized(B_kolona_v, 0, 1 * sizeof(int), &B_kolona);
    MPI_Type_commit(&B_kolona);

    MPI_Scatter(B, 1, B_kolona, locB, S * kolona_po_procesu, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < kolona_po_procesu; j++)
            for (int k = 0; k < S; k++)
                locC[i * kolona_po_procesu + j] += A[i][k] * locB[k * kolona_po_procesu + j];

    // #2
    for (int i = 0; i < S; i++)
    {
        locProdB[i] = 1;
        for (int j = 0; j < kolona_po_procesu; j++)
            locProdB[i] *= locB[i * kolona_po_procesu + j];
    }

    // #3
    MPI_Datatype C_kolona_v, C_kolona;
    MPI_Type_vector(N * kolona_po_procesu, 1, p, MPI_INT, &C_kolona_v);
    MPI_Type_create_resized(C_kolona_v, 0, 1 * sizeof(int), &C_kolona);
    MPI_Type_commit(&C_kolona);

    MPI_Gather(locC, N * kolona_po_procesu, MPI_INT, C, 1, C_kolona, 0, MPI_COMM_WORLD);
    MPI_Reduce(locProdB, prodB, S, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Matrica C:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
                printf("%d ", C[i][j]);
            printf("\n");
        }

        printf("Proizvod kolona matrice B:\n");
        for (int i = 0; i < S; i++)
            printf("%d ", prodB[i]);
        printf("\n");
    }

    free(locB);
    free(locC);
    MPI_Finalize();
}

/*

    #1 - pravimo tip da svako dobije odgovarjuce kolone sa pravilim skokovim (opisano u tekstu)
    #2 - svako nepotpuni rezultat za svaku kolonu
    #3 - posto i rezultat je podeljen kao kolone koje se primali, mora da se napravi tip koji ce da
        osigura da se i rezultat upakuje kako treba

*/