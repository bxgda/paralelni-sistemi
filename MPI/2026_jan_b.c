/*

   Napisati MPI program koji će omogućiti komunikaciju samo procesa na glavnoj dijagonali procesa u kvadratnoj mreži
   procesa kreiranjem komunikatora. Master proces (P0) svim procesima ove grupe šalje po jednu kolonu matrice A.
   Prikazati identifikatore procesa koji pripadaju novom komunikatoru, kao i proizvod primljene kolone u svakom procesu
   komunikatora.

*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2 // dimenzija matrice A

int main(int argc, char **argv)
{
    int A[N][N], kolona[N];
    int rank, p;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // #1
    if (rank == 0)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = i + j + 1;

    // #2
    int q = (int)sqrt(p); // dimenzija matrice procesa
    int red_procesa = rank / q;
    int kolona_procesa = rank % q;

    // #3
    MPI_Comm dijagonala;
    int color = 0;

    if (red_procesa == kolona_procesa)
        color = 1;

    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &dijagonala);

    // #4
    MPI_Datatype tip_kolona_v, tip_kolona;
    MPI_Type_vector(N, 1, N, MPI_INT, &tip_kolona_v);
    MPI_Type_create_resized(tip_kolona_v, 0, sizeof(int), &tip_kolona);
    MPI_Type_commit(&tip_kolona);

    if (color == 1)
    {
        // #5
        int novi_rank, novi_p;
        MPI_Comm_rank(dijagonala, &novi_rank);
        MPI_Comm_size(dijagonala, &novi_p);

        // #6
        if (novi_rank == 0)
        {
            for (int i = 1; i < N; i++)
                MPI_Send(&A[0][i], 1, tip_kolona, i, 10, dijagonala);

            for (int i = 0; i < N; i++)
                kolona[i] = A[i][0];
        }
        else
            MPI_Recv(kolona, N, MPI_INT, 0, 10, dijagonala, &status);

        // #7
        int proizvod_kolone = 1;
        for (int i = 0; i < N; i++)
            proizvod_kolone *= kolona[i];

        printf("Novi rank: %d, proizvod kolone: %d\n", novi_rank, proizvod_kolone);
    }

    MPI_Finalize();
}

/*

    #1 - master proces (P0) inicijalizuje matricu A
    #2 - svaki proces odredjuje svoj red i kolonu u kvadratnoj mrezi procesa
    #3 - kreira se novi komunikator koji sadrzi samo procese na glavnoj dijagonali
    #4 - kreira se novi tip podatka koji predstavlja kolonu matrice A
    #5 - racunamo nove rankove i broj procesa u novom komunikatoru
    #6 - master proces (novi rank 0) salje odgovarajuce kolone matrice A procesima u novom komunikatoru, a on sam prima svoju kolonu (samo je pretumbamo) 
    #7 - svaki proces racuna proizvod primljene kolone i stampa svoj novi rank i proizvod

*/
