/*

  1. zadatak: Napisati MPI program koji realizuje množenje matrice \( A_{k \times n} \) i vektora \( b_n \). Matrica i
  vektor se inicijalizuju u master procesu. Matrica je podeljena u blokove po vrstama tako da će proces \( P_i \) dobiti
  prvih \( 2^i \) vrsta, proces \( P_{i+1} \) dobije sledećih \( 2^{i+1} \) vrsta, itd. Vektor \( b \) se u celosti
  šalje svim procesima. Predvideti da se slanje blokova matrica svakom procesu šalje jednim MPI_Send pozivom kojim se
  šalju svi neophodni elementi matrice, dok se slanje vektora \( b \) obavlja grupnim operacijama. Svaki proces obavlja
  odgovarajuća izračunavanja i učestvuje u generisanju rezultata. Rezultujući vektor \( d \) treba se naći u procesu
  koji je učitao najviše vrsta matrice \( A \). Dati primer matrice \( A \) i vektora \( b \) i ilustrovati podelu
  podataka po procesima, kao i izgled rezultata za izabran broj procesa. Napisati na koji se način iz komandne linije
  vrši startovanje napisane MPI aplikacije. [45 poena]

*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3
#define K 15 // 4 procesa: 1, 2, 4, 8 vrsta matrice A

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

    // #1
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

    // #2
    for (int i = 0; i < svoji_redovi; i++)
        for (int j = 0; j < N; j++)
            locC[i] += locA[i * N + j] * B[j];

    // #3
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

/*

    #1 saljemo vektor B svima preko Bcast-a a vrste saljemo preko send i recv jer su blokovi razlicite velicine a i tako se trazi u zadatku 
    #2 svaki proces racuna svoj deo rezultata #3 rezultujuci vektor C treba da se nalazi u procesu
     koji je ucitao najvise (to je uvek poslednji proces) i onda ostali salju a on prima po tacnom redosledu i onda stampa rezultat

*/
