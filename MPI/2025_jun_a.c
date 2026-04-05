/*
    TEKST ZADATKA:

    Napisati MPI program koji realizuje množenje matrice Aₘₓₖ i vektora bₖ.
    Takođe, program kao rezultat treba da generiše i proizvod elemenata svake kolone matrice A.
    Matrica A i vektor b se inicijalizuju u master procesu. Matrica A je distribuirana u blokove po vrstama
    (p-broj procesa, k je deljivo sa p) i to tako da proces Pᵢ dobija vrste sa indeksima l, l mod p = i (0 ≤ i ≤ p-1),
    tj. vrste sa indeksima i, i+p, i+2p, …, i+m-p. Master proces distribuira odgovarajuće vrste matrice A (direktno iz matrice)
    i ceo vektor b svim procesima. Predvideti da se slanje svih vrsta matrice A jednom procesu obavlja odjednom.
    Predvideti da se slanje elemenata vektora b jednom procesu obavlja odjednom.
    Svaki proces obavlja odgovarajuća izračunavanja i učestvuje u generisanju rezultata.
    Zadatak rešiti korišćenjem grupnih operacija.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 8
#define K 4

int main(int argc, char *argv[])
{
    // #1
    int A[M][K], B[K], C[M], prod[K];
    int rank, p;

    // #2
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // #3
    int redovi_po_procesu = M / p;

    int *loc_A = (int *)malloc(redovi_po_procesu * K * sizeof(int));
    int *loc_C = (int *)calloc(redovi_po_procesu, sizeof(int));
    int *loc_prod = (int *)malloc(K * sizeof(int));

    // #4
    if (rank == 0)
    {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                A[i][j] = i + j + 1;

        for (int i = 0; i < K; i++)
            B[i] = 1;
    }

    // #5
    MPI_Datatype redTip_v, redTip;
    MPI_Type_vector(redovi_po_procesu, K, K * p, MPI_INT, &redTip_v);
    MPI_Type_create_resized(redTip_v, 0, K * sizeof(int), &redTip);
    MPI_Type_commit(&redTip);

    // #6
    MPI_Scatter(A, 1, redTip, loc_A, redovi_po_procesu * K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K, MPI_INT, 0, MPI_COMM_WORLD);

    // #7
    for (int i = 0; i < redovi_po_procesu; i++)
        for (int j = 0; j < K; j++)
            loc_C[i] += loc_A[i * K + j] * B[j];

    for (int i = 0; i < K; i++)
    {
        loc_prod[i] = 1;
        for (int j = 0; j < redovi_po_procesu; j++)
            loc_prod[i] *= loc_A[j * K + i];
    }

    // #8
    MPI_Gather(loc_C, redovi_po_procesu, MPI_INT, C, redovi_po_procesu, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Reduce(loc_prod, prod, K, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

    // #9
    if (rank == 0)
    {
        printf("vektor C = ");
        for (int i = 0; i < M; i++)
            printf("%d ", C[i]);
        printf("\nproizvod svake kolone: ");
        for (int i = 0; i < K; i++)
            printf("%d ", prod[i]);
        printf("\n");
    }

    // #10
    free(loc_A);
    free(loc_C);
    free(loc_prod);

    MPI_Finalize();
}

/*
    #1 - staticko alociranje memorije za promenljive za koje unapred znamo koliko ce da zauzimaju
    #2 - mpi inicijalizacija
    #3 - izracunavanje koliko ce svaki red da dobije kolona i na osnovu toga dinamicko alociranje memorije za lokalne promenljive
    #4 - inicijalizacija matrice i vektora u procesu 0
    #5 - posto se vrse ne salju po blokovima nego po navedenom principu, moramo da napravimo poseban tip za takvu distribuciju
    #6 - distribucija redova tako da svaki proces dobije svoj par vrsta i distribucija vektora svima
    #7 - svako racuna svoj deo za rezultujuci vektor (svaki proces potpuno izracunava "redova_po_procesu" delova vektora)
         svaki proces ucestvuje u delu proizvoda svake kolone (svaki proces izracuna parcijalni proizvod za "redova_po_procesu" elemenata svake kolone)
    #8 - sakupljamo rezultate, gather za vektor i reduce (mpi_sum) za rezultat proizvoda svake kolone
    #9 - proces 0 ispisuje rezultate
    #10 - necemo mem-leak
*/