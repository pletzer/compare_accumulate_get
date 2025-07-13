#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank + 1; // Each rank contributes its rank + 1
    int result_accumulate = 0;
    int result_get = 0;

    MPI_Win win;
    MPI_Win_create(&result_accumulate, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // --- MPI_Accumulate ---
    MPI_Win_fence(0, win);
    double start_acc = MPI_Wtime();
    if (rank != 0) {
        MPI_Accumulate(&local_value, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
    }
    MPI_Win_fence(0, win);
    double end_acc = MPI_Wtime();

    if (rank == 0) {
        result_accumulate += local_value; // Add own value
        std::cout << "MPI_Accumulate result: " << result_accumulate
                  << ", Time: " << (end_acc - start_acc) << " seconds\n";
    }

    // --- MPI_Get ---
    std::vector<int> all_values(size, 0);
    MPI_Win win_get;
    MPI_Win_create(&local_value, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_get);
    MPI_Win_fence(0, win_get);

    double start_get = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            MPI_Get(&all_values[i], 1, MPI_INT, i, 0, 1, MPI_INT, win_get);
        }
    }
    MPI_Win_fence(0, win_get);
    double end_get = MPI_Wtime();

    if (rank == 0) {
        result_get = std::accumulate(all_values.begin(), all_values.end(), 0);
        std::cout << "MPI_Get result: " << result_get
                  << ", Time: " << (end_get - start_get) << " seconds\n";
    }

    MPI_Win_free(&win);
    MPI_Win_free(&win_get);
    MPI_Finalize();

    return 0;

}


