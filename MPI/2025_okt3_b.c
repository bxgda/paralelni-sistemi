/*

    Napisati MPI program koji će omogućiti komunikaciju samo procesa u donjoj trougaonoj matrici u
    kvadratnoj mreži procesa kreiranjem komunikatora. Obezbediti da svi procesi ove grupe dobiju sve
    elemente glavne dijagonale matrice A, direktno iz matrice.

*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4

int main(int argc, char **argv)
{
    int A[N][N], dijagonala[N];
    int rank, p;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (rank == 0)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = i + j + 1;

    int q = (int)sqrt(p);
    int red_procesa = rank / q;
    int kolona_procesa = rank % q;

    int boja = 0;
    if (red_procesa >= kolona_procesa)
        boja = 1;

    MPI_Comm donji_trougao;
    MPI_Comm_split(MPI_COMM_WORLD, boja, rank, &donji_trougao);

    MPI_Datatype tip_dijagonala;
    MPI_Type_vector(N, 1, N + 1, MPI_INT, &tip_dijagonala);
    MPI_Type_commit(&tip_dijagonala);

    if (boja == 1)
    {
        int novi_rank, novi_p;
        MPI_Comm_rank(donji_trougao, &novi_rank);
        MPI_Comm_size(donji_trougao, &novi_p);

        if (novi_rank == 0)
        {
            for (int i = 1; i < novi_p; i++)
                MPI_Send(&A[0][0], 1, tip_dijagonala, i, 10, donji_trougao);

            for (int i = 0; i < N; i++)
                dijagonala[i] = A[i][i];
        }
        else
            MPI_Recv(dijagonala, N, MPI_INT, 0, 10, donji_trougao, &status);

        printf("Globalni rank: %d (Novi: %d) primio dijagonalu: ", rank, novi_rank);
        for (int i = 0; i < N; i++)
            printf("%d ", dijagonala[i]);
        printf("\n");
    }

    MPI_Finalize();
}