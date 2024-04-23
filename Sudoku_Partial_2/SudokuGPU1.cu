
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>

#define SIZE 9

__device__ bool isSafe(char* board, int row, int col, char num) {
    // Verifica si el número ya está en la fila
    //bool rowSet[SIZE] = { false };
    for (int i = 0; i < SIZE; i++) {
        if (board[row * SIZE + i] != '.' && board[row * SIZE + i] == num) {
            return false;
        }
    }

    // Verifica si el número ya está en la columna
    //bool colSet[SIZE] = { false };
    for (int i = 0; i < SIZE; i++) {
        if (board[i * SIZE + col] != '.' && board[i * SIZE + col] == num) {
            return false;
        }
    }

    // Verifica si el número ya está en el subcuadro 3x3
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    //bool subgridSet[SIZE] = { false };
    for (int i = startRow; i < startRow + 3; i++) {
        for (int j = startCol; j < startCol + 3; j++) {
            if (board[i * SIZE + j] != '.' && board[i * SIZE + j] == num) {
                return false;
            }
        }
    }

    return true;
}


__global__ void solveSudoku(char* board, bool* solved, bool* found, int idx) {
    /*int idx = threadIdx.x + blockIdx.x * blockDim.x; */
    //idx != 0 ? printf(" threadNum: %d, board value: %c, prev board: %c \n", idx, board[idx], board[idx - 1]) : printf(" threadNum: %d, board value: %c \n", idx, board[idx]);
    if (*solved && *found && idx >= SIZE * SIZE) return;

    int row = idx / SIZE;
    int col = idx % SIZE;


    if (board[idx] == '.') {
        for (char num = '1'; num <= '9'; num++) {
            if (isSafe(board, row, col, num)) {
                board[idx] = num;
                /* if (idx == SIZE * SIZE - 1) {
                     *solved = true;
                     *found = true;
                 }*/

                solveSudoku <<<1, 1>>> (board, solved, found, idx + 1);
                __syncthreads();
                if (*solved && *found) return;

                board[idx] = '.';
            }
        }
    }
    else {
        solveSudoku <<<1, 1 >>> (board, solved, found, idx + 1);
        __syncthreads();
        if (idx == SIZE * SIZE - 1) return;
    }
    
    if (idx >= SIZE * SIZE) {
        *solved = true;
        *found = true;
        return;
    }
}

int main() {
    /*
    char board[SIZE][SIZE] = {
        {'5','3','.','.','7','.','.','.','2'},
        {'6','.','.','1','9','5','.','.','.'},
        {'.','9','8','.','.','.','7','6','.'},
        {'8','.','.','.','6','.','.','.','3'},
        {'4','.','.','8','.','3','.','.','1'},
        {'7','.','.','.','2','.','.','.','6'},
        {'.','6','.','.','.','.','2','8','.'},
        {'.','.','.','4','1','9','.','.','5'},
        {'.','8','.','.','8','.','.','7','9'}
    };
    */

    char board[SIZE][SIZE] = {
        {'.', '.', '7', '8', '6', '.', '2', '.', '1'},
        {'6', '1', '.', '.', '.', '.', '.', '.', '.'},
        {'8', '.', '2', '4', '1', '3', '.', '.', '.'},
        {'2', '.', '4', '.', '.', '5', '.', '.', '9'},
        {'9', '8', '.', '.', '3', '.', '7', '.', '.'},
        {'.', '.', '3', '.', '.', '8', '.', '.', '2'},
        {'.', '.', '.', '3', '8', '.', '1', '9', '7'},
        {'.', '3', '.', '.', '.', '.', '5', '.', '4'},
        {'1', '9', '.', '.', '.', '.', '6', '.', '3'}
    };

    char* dev_board;
    bool* dev_solved;
    bool* dev_found;
    bool solved = false;
    bool found = false;

    cudaMalloc((void**)&dev_board, SIZE * SIZE * sizeof(char));
    cudaMalloc((void**)&dev_solved, sizeof(bool));
    cudaMalloc((void**)&dev_found, sizeof(bool));

    cudaMemcpy(dev_board, board, SIZE * SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_solved, &solved, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_found, &found, sizeof(bool), cudaMemcpyHostToDevice);

    solveSudoku << <1, 1 >> > (dev_board, dev_solved, dev_found, 0);
    
    cudaMemcpy(&solved, dev_solved, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(board, dev_board, SIZE * SIZE * sizeof(char), cudaMemcpyDeviceToHost);

    if (solved) {
        std::cout << "El Sudoku resuelto es:" << std::endl;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                std::cout << board[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        std::cout << "No se encontró una solución válida para el Sudoku." << std::endl;
    }

    cudaFree(dev_board);
    cudaFree(dev_solved);
    cudaFree(dev_found);

    return 0;
}