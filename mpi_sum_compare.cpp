#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <fstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <size1> <size2> ...\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::ofstream csvfile;
    if (rank == 0) {
	csvfile.open("timings.csv");
        csvfile << "method,data_size,time_sec,result_sum\n";
    }

    for (int i = 1; i < argc; ++i) {

        size_t data_size = std::stoul(argv[i]);

        std::vector<double> local_data(data_size, rank + 1.0);
        std::vector<double> result_accumulate(data_size, 0.0);
        std::vector<double> result_get(size, 0.0);  // one value per rank

        // --- MPI_Accumulate ---
        MPI_Win win_acc;
        MPI_Win_create(result_accumulate.data(), data_size * sizeof(double), sizeof(double),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_acc);

        MPI_Barrier(MPI_COMM_WORLD);
        double start_acc = MPI_Wtime();

        MPI_Win_fence(0, win_acc);
        if (rank != 0) {
            MPI_Accumulate(local_data.data(), data_size, MPI_DOUBLE,
                           0, 0, data_size, MPI_DOUBLE, MPI_SUM, win_acc);
        }
        MPI_Win_fence(0, win_acc);
        double end_acc = MPI_Wtime();

        if (rank == 0) {
            for (size_t j = 0; j < data_size; ++j) {
                result_accumulate[j] += local_data[j];  // add own value
            }
            double sum = std::accumulate(result_accumulate.begin(), result_accumulate.end(), 0.0);
            csvfile << "accumulate," << data_size << "," << (end_acc - start_acc) << "," << sum << "\n";
        }

        MPI_Win_free(&win_acc);

        // --- MPI_Get ---
        MPI_Win win_get;
        MPI_Win_create(local_data.data(), data_size * sizeof(double), sizeof(double),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_get);
        MPI_Barrier(MPI_COMM_WORLD);

        double start_get = MPI_Wtime();
        std::vector<double> gathered(data_size * size, 0.0);
        MPI_Win_fence(0, win_get);
        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                MPI_Get(&gathered[r * data_size], data_size, MPI_DOUBLE,
                        r, 0, data_size, MPI_DOUBLE, win_get);
            }
        }
        MPI_Win_fence(0, win_get);
        double end_get = MPI_Wtime();

        if (rank == 0) {
            double sum = std::accumulate(gathered.begin(), gathered.end(), 0.0);
            csvfile << "get," << data_size << "," << (end_get - start_get) << "," << sum << "\n";
        }

        MPI_Win_free(&win_get);
    }

    MPI_Finalize();

    return 0;

}

