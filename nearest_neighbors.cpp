#include <vector>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <typeinfo>


//The provided read_xyz_file function:
//-------------------------------------------
// function to read in a list of 3D coordinates from an .xyz file
// input: the name of the file
std::vector < std::vector < double > > read_xyz_file(std::string filename, int& N, double& L){

  // open the file
  std::ifstream xyz_file(filename);

  // read in the number of atoms
  xyz_file >> N;
  
  // read in the cell dimension
  xyz_file >> L;
  
  // now read in the positions, ignoring the atomic species
  std::vector < std::vector < double > > positions;
  std::vector < double> pos = {0, 0, 0};
  std::string dummy; 
  for (int i=0;i<N;i++){
    xyz_file >> dummy >> pos[0] >> pos[1] >> pos[2];
    positions.push_back(pos);           
  }
  
  // close the file
  xyz_file.close();
  
  return positions;
  
}
//-------------------------------------------


//This provides a unique index for each cell used in the cell list model,
//allowing cells to be stored in a 1D vector encoded by their 3D coordinate
int calcIndex(int i, int j, int k, int N_cells){
  return i + j*N_cells + k*N_cells*N_cells;
}


//Used to print the vector of neighbours to data files
void printVector(const std::vector<double>& vec) {
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
}


//Single threaded implementation of the nearest neighbor algorithm.
//Double increment refers to the optimization technique of adjusting the neighbour list of atom 1 and atom 2 in the loop
int bruteforce_double_increment(int argc, char **argv)
{
    //We only use MPI here to track the start and end time
    MPI_Init(&argc, &argv);
    // read in the name of the file
    std::string filename = argv[1];
    //0 or 1 to print results
    int printResults = atoi(argv[3]);
    //0 or 1 to print time
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;
    // read in the positions
    int N;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);

    //-------------------------------------------

    double start = MPI_Wtime();
    //For each atom, we loop over all other atoms and calculate the distance
    for (int i = 0; i < N; i++) {    
        currentNeighbours = 0;
        for (int j = i+1; j < N; j++) {
          if (i == j) continue;
            double mag_r_sq = 0.0;
            std::vector < double > ri = positions[i]; //Current atom
            std::vector < double > rj = positions[j]; //Current neighbor
            mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
            if (mag_r_sq < rc_sq) {
                neighbours[i]++;
                neighbours[j]++;
            }
        }
    }
    double finish = MPI_Wtime();
    MPI_Finalize();

    //-------------------------------------------

    if (printTime == 1){
      std::cout << finish - start << ",";
    }
    if (printResults == 1){
      printVector(neighbours);
    }
    return 0;

}


//Single threaded implementation of the nearest neighbor algorithm.
int bruteforce_square_optimized(int argc, char **argv)
{
    //Only use MPI to track the start and end time; the algorithm is single threaded
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int printResults = atoi(argv[3]);
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;

    // read in the positions
    int N;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);

    //-------------------------------------------

    double start = MPI_Wtime();
    // loop over all pairs of atoms
    for (int i = 0; i < N; i++) {    
        currentNeighbours = 0;
        for (int j = 0; j < N; j++) {
          if (i == j) continue; 
            double mag_r_sq = 0.0;
            std::vector < double > ri = positions[i];
            std::vector < double > rj = positions[j];
            mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
            //If the distance is less than the cutoff radius, we have a neighbour
            if (mag_r_sq < rc_sq) {
                currentNeighbours++;
            }
        }
        neighbours[i] = currentNeighbours;
    }
    double finish = MPI_Wtime();
    MPI_Finalize();

    //-------------------------------------------
    
    if (printTime == 1){
      std::cout << finish - start << ",";
    }
    if (printResults == 1){
      printVector(neighbours);
    }
    return 0;
}


//MPI implementation of the nearest neighbor algorithm using a cyclic distribution of atoms to threads
int MPI_cyclic_Bruteforce(int argc, char **argv)
{
    //MPI setup
    MPI_Init(&argc, &argv);
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int procID;
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    std::string filename = argv[1];
    int printResults = atoi(argv[3]);
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;
    
    // read in the positions
    int N;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);

    
    double start = MPI_Wtime();
    //Round robin/cyclic distribution of atoms to threads
    //Each thread is responsible for a subset of the atoms with indecies procID, procID+numProcs, procID+2*numProcs, ...
    for (int i = 0+procID; i < N; i+=numProcs) {    
        currentNeighbours = 0;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double mag_r_sq = 0.0;
            std::vector < double > ri = positions[i];
            std::vector < double > rj = positions[j];
            //Optimization, we do not need to squareroot the distance for every atom,
            //We can just square rc once and compare it to the squared distance
            mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
            //If the distance is less than the cutoff radius, we have a neighbour
            if (mag_r_sq < rc_sq) {
                currentNeighbours++;
            }
        }
        neighbours[i] = currentNeighbours;
    }
    double finish = MPI_Wtime();

    //-------------------------------------------
    
    //We must collapse the neighbours vector to the root process
    if (procID == 0) {
      // If the process ID is 0 (the root process),
      // Instantiate a temporary vector to store the neighbor list from other processes
      std::vector<double> tempNeighbours = std::vector<double>(N, 0.0);
      MPI_Status status;

      // Loop through all the other processes (starting from 1)
      for (int i = 1; i < numProcs; i++) {
        // Receive data from process i and store it in the temporary vector
        MPI_Recv(tempNeighbours.data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

        // Add the received data to the main 'neighbours' vector
        for (int j = 0; j < N; j++) {
          neighbours[j] += tempNeighbours[j];
        }
      }
    } else {
      // For all other processes (not the root process),
      // send the 'neighbours' vector data to the root process (process 0)
      MPI_Send(neighbours.data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    //-------------------------------------------

    if (procID == 0) {
        if (printTime == 1){
            std::cout << finish - start << ",";
        }
        if (printResults == 1){
            printVector(neighbours);
        }
    }
    return 0;
}


//MPI implementation of the nearest neighbor algorithm using a chunked distribution of atoms to threads
int MPI_chunk_bruteforce(int argc, char **argv)
{
    //MPI setup
    MPI_Init(&argc, &argv);
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int procID;
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);
    double start = MPI_Wtime();

    std::string filename = argv[1];
    int printResults = atoi(argv[3]);
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;
    int N = 0;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);
    
    // Hacky calculation to round up and get chunked index
    int chunkSeperation = (N + numProcs - 1) / numProcs;

    // We are distributing the threads accross contiguous chunks of the array
    for (int i = procID*chunkSeperation; i < (procID+1)*chunkSeperation && i<N; i+=1) {    
        currentNeighbours = 0;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double mag_r_sq = 0.0;
            std::vector < double > ri = positions[i];
            std::vector < double > rj = positions[j];
            mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
            if (mag_r_sq < rc_sq) {
              currentNeighbours++;
            }
            }
            neighbours[i] = currentNeighbours;
            }
    double finish = MPI_Wtime();

    //-------------------------------------------

    //Collapse the neighbours vector to the root process
    if (procID == 0) {
        // If the process ID is 0 (the root process),
        // create a temporary vector to store the received data from other processes
        std::vector<double> tempNeighbours = std::vector<double>(N, 0.0);
        MPI_Status status;

        // Loop through all the other processes (starting from 1)
        for (int i = 1; i < numProcs; i++) {
            // Receive data from process i and store it in the temporary vector
            MPI_Recv(tempNeighbours.data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            // Add the received data to the main 'neighbours' vector
            for (int j = 0; j < N; j++) {
                neighbours[j] += tempNeighbours[j];
            }
        }
    } 
    else {
        // For all other processes (not the root process),
        // send the 'neighbours' vector data to the root process (process 0)
        MPI_Send(neighbours.data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    //-------------------------------------------

    if (procID == 0)
    {
        if (printTime == 1){
            std::cout << finish - start << ",";
        }
        if (printResults == 1){
            printVector(neighbours);
        }
    }
    return 0;
    }

//Single threaded implementation of the nearest neighbor algorithm using a cell list model
//This is a more efficient algorithm as it reduces the number of comparisons
int cell_list(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int printResults = atoi(argv[3]);
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;
    // read in the positions
    int N;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    int N_cells = std::ceil(L/rc);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);
    //1D vector containing of cells containing an int vector of the cell index
    std::vector < std::vector < int > > cell_list = std::vector < std::vector < int > > (N_cells*N_cells*N_cells, std::vector < int > ());
    //Store a 2D vector of dimensions N_cells^3 x N_cells^3 containing a boolean value, 
    //indicating if the cell pair has been checked
    std::vector < std::vector < bool > > cell_checked = std::vector < std::vector < bool > > (N_cells*N_cells*N_cells, std::vector < bool > (N_cells*N_cells*N_cells, false));

    double start = MPI_Wtime();

    // Loop over all atoms and assign them to a cell
    for (int i = 0; i < N; i++) {
        int cell_x = std::floor(positions[i][0]/(rc+0.001));
        int cell_y = std::floor(positions[i][1]/(rc+0.001));
        int cell_z = std::floor(positions[i][2]/(rc+0.001));
        int cell_index = calcIndex(cell_x, cell_y, cell_z, N_cells);
        if (cell_index >= 0 && cell_index < cell_list.size()) {
            cell_list[cell_index].push_back(i);
        }
    }

    // Loop over all cells ix, iy, iz
    // Loop over all neighbouring cells jx, jy, jz
    // Loop over all atoms in cell ix, iy, iz
    // Loop over all atoms in cell jx, jy, jz

    for (int ix = 0; ix < N_cells; ix++) {
        for (int iy = 0; iy < N_cells; iy++) {
            for (int iz = 0; iz < N_cells; iz++) {
                // Loop over all neighbouring cells
                for (int jx = ix -1; jx < ix+2; jx++) {
                    for (int jy = iy -1; jy < iy+2; jy++) {
                        for (int jz = iz -1; jz < iz+2; jz++) {
                            if (jx < 0 || jx >= N_cells || jy < 0 || jy >= N_cells || jz < 0 || jz >= N_cells) {
                            continue;
                            }
                            // if (cell_checked[ix + iy*N_cells + iz*N_cells*N_cells][jx + jy*N_cells + jz*N_cells*N_cells]) {
                            //   continue;
                            // }
                            int cell_index = calcIndex(ix, iy, iz, N_cells);
                            int neighbour_index = calcIndex(jx, jy, jz, N_cells);
                            for (int i = 0; i < cell_list[cell_index].size(); i++) {
                                for (int j = 0; j < cell_list[neighbour_index].size(); j++) {
                                    if (cell_index==neighbour_index && i == j) {
                                        continue;
                                    }
                                    double mag_r_sq = 0.0;
                                    std::vector < double > ri = positions[cell_list[cell_index][i]];
                                    std::vector < double > rj = positions[cell_list[neighbour_index][j]];
                                    mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
                                    if (mag_r_sq < rc_sq) {
                                        neighbours[cell_list[cell_index][i]]++;
                                    }
                                }
                            }
                            cell_checked[ix + iy*N_cells + iz*N_cells*N_cells][jx + jy*N_cells + jz*N_cells*N_cells] = true;
                            cell_checked[jx + jy*N_cells + jz*N_cells*N_cells][ix + iy*N_cells + iz*N_cells*N_cells] = true;
                        }
                    }
                }
            }
        }
    }

    //-------------------------------------------

    double finish = MPI_Wtime();
    MPI_Finalize();
    if (printTime == 1){
        std::cout << finish - start << ",";
    }
    if (printResults == 1){
        printVector(neighbours);
    }
    return 0;
}

int MPI_cell_list(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int procID;
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);
    
    std::string filename = argv[1];
    int printResults = atoi(argv[3]);
    int printTime = atoi(argv[4]);
    double rc = 9.0;
    double rc_sq = rc*rc;
    int N;
    double L;
    int currentNeighbours;
    std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    int N_cells = std::ceil(L/rc);
    std::vector < double > neighbours = std::vector < double > (N, 0.0);
    //1D vector containing of cells containing an int vector of the cell index
    std::vector < std::vector < int > > cell_list = std::vector < std::vector < int > > (N_cells*N_cells*N_cells, std::vector < int > ());
    //Store a 2D vector of dimensions N_cells^3 x N_cells^3 containing a boolean value 
    //indicating if the cell pair has been checked
    std::vector < std::vector < bool > > cell_checked = std::vector < std::vector < bool > > (N_cells*N_cells*N_cells, std::vector < bool > (N_cells*N_cells*N_cells, false));


    double start = MPI_Wtime();
    // Loop over all atoms and assign them to a cell
    for (int i = 0; i < N; i++) {
        int cell_x = std::floor(positions[i][0]/(rc+0.001));
        int cell_y = std::floor(positions[i][1]/(rc+0.001));
        int cell_z = std::floor(positions[i][2]/(rc+0.001));
        int cell_index = calcIndex(cell_x, cell_y, cell_z, N_cells);
        if (cell_index >= 0 && cell_index < cell_list.size()) {
            cell_list[cell_index].push_back(i);
        }
    }

    // Loop over all cells ix, iy, iz
    // Loop over all neighbouring cells jx, jy, jz
    // Loop over all atoms in cell ix, iy, iz
    // Loop over all atoms in cell jx, jy, jz

    for (int ix = 0; ix < N_cells; ix++) {
        for (int iy = 0; iy < N_cells; iy++) {
            for (int iz = 0; iz < N_cells; iz++) {
            // Loop over all neighbouring cells
                for (int jx = ix -1; jx < ix+2; jx++) {
                    for (int jy = iy -1; jy < iy+2; jy++) {
                        for (int jz = iz -1; jz < iz+2; jz++) {
                            if (jx < 0 || jx >= N_cells || jy < 0 || jy >= N_cells || jz < 0 || jz >= N_cells) {
                            continue;
                            }
                            if (cell_checked[ix + iy*N_cells + iz*N_cells*N_cells][jx + jy*N_cells + jz*N_cells*N_cells]) {
                              continue;
                            }
                            int cell_index = calcIndex(ix, iy, iz, N_cells);
                            int neighbour_index = calcIndex(jx, jy, jz, N_cells);
                            for (int i = 0+procID; i < cell_list[cell_index].size(); i+=numProcs) {
                                for (int j = 0; j < cell_list[neighbour_index].size(); j++) {
                                    if (cell_index==neighbour_index && i == j) {
                                        continue;
                                    }
                                    double mag_r_sq = 0.0;
                                    std::vector < double > ri = positions[cell_list[cell_index][i]];
                                    std::vector < double > rj = positions[cell_list[neighbour_index][j]];
                                    mag_r_sq = pow(ri[0]-rj[0],2) + pow(ri[1]-rj[1],2) + pow(ri[2]-rj[2],2);
                                    if (mag_r_sq < rc_sq) {
                                        neighbours[cell_list[cell_index][i]]++;
                                    }
                                }
                            }
                            cell_checked[ix + iy*N_cells + iz*N_cells*N_cells][jx + jy*N_cells + jz*N_cells*N_cells] = true;
                            cell_checked[jx + jy*N_cells + jz*N_cells*N_cells][ix + iy*N_cells + iz*N_cells*N_cells] = true;
                        }
                    }
                }
            }
        }
    }


    double finish = MPI_Wtime();

    if (procID == 0) {
    // If the process ID is 0 (the root process),
    // create a temporary vector to store the received data from other processes
    std::vector<double> tempNeighbours = std::vector<double>(N, 0.0);
    MPI_Status status;

    // Loop through all the other processes (starting from 1)
    for (int i = 1; i < numProcs; i++) {
        // Receive data from process i and store it in the temporary vector
        MPI_Recv(tempNeighbours.data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

        // Add the received data to the main 'neighbours' vector
        for (int j = 0; j < N; j++) {
        neighbours[j] += tempNeighbours[j];
        }
    }
    } else {
    // For all other processes (not the root process),
    // send the 'neighbours' vector data to the root process (process 0)
    MPI_Send(neighbours.data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    if (procID == 0)
    {
    if (printTime == 1){
        std::cout << finish - start << ",";
    }
    if (printResults == 1){
        printVector(neighbours);
    }
    }
    return 0;
}

int main(int argc, char **argv)
{

    int Method = atoi(argv[2]);  

    if (Method == 1){
        bruteforce_square_optimized(argc, argv);
    }

    else if (Method == 2){
        MPI_cyclic_Bruteforce(argc, argv);
    }

    else if (Method == 3){
        MPI_chunk_bruteforce(argc, argv);
    }

    else if (Method == 4){
        cell_list(argc, argv);
    }

    else if (Method == 5){
        MPI_cell_list(argc, argv);
    }
    else if (Method == 6){
        bruteforce_double_increment(argc, argv);
    }

    else {
        std::cout << "Invalid Method" << std::endl;
    }

    return 0;
}
