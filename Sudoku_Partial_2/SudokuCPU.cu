#include <iostream>
#include <vector>
#include <cassert>

class ValidateSudoku {
private:
    std::vector<std::vector<int>> tablero;
    std::vector<int> lista_invertida;

public:
    ValidateSudoku(std::vector<std::vector<int>> tab) : tablero(tab) {}

    void chequeo_general() {
        // Chequear que el tablero introducido sea un tablero 9x9
        assert(tablero.size() == 9 && "El tablero ingresado no respeta el formato 9x9");
        for (const auto& fila : tablero) {
            assert(fila.size() == 9 && "El tablero ingresado no respeta el formato 9x9");
        }
    }

    void print_board(const std::vector<std::vector<int>>& board) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                std::cout << board[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    bool find_empty_location(const std::vector<std::vector<int>>& board, std::vector<int>& l) {
        for (int row = 0; row < 9; ++row) {
            for (int col = 0; col < 9; ++col) {
                if (board[row][col] == 0) {
                    l[0] = row;
                    l[1] = col;
                    return true;
                }
            }
        }
        return false;
    }

    bool used_in_row(const std::vector<std::vector<int>>& board, int row, int num) {
        for (int i = 0; i < 9; ++i) {
            if (board[row][i] == num) {
                return true;
            }
        }
        return false;
    }

    bool used_in_col(const std::vector<std::vector<int>>& board, int col, int num) {
        for (int i = 0; i < 9; ++i) {
            if (board[i][col] == num) {
                return true;
            }
        }
        return false;
    }

    bool used_in_box(const std::vector<std::vector<int>>& board, int row, int col, int num) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (board[i + row][j + col] == num) {
                    return true;
                }
            }
        }
        return false;
    }

    bool is_safe_location(const std::vector<std::vector<int>>& board, int row, int col, int num) {
        return !used_in_row(board, row, num) && !used_in_col(board, col, num) && !used_in_box(board, row - row % 3, col - col % 3, num);
    }

    bool solve_sudoku(std::vector<std::vector<int>>& board) {
        std::vector<int> l(2, 0);

        if (!find_empty_location(board, l)) {
            return true;
        }

        int row = l[0];
        int col = l[1];

        for (int num = 1; num <= 9; ++num) {
            if (is_safe_location(board, row, col, num)) {
                board[row][col] = num;

                if (solve_sudoku(board)) {
                    return true;
                }

                board[row][col] = 0;
            }
        }

        return false;
    }
};

int main() {

    std::vector<std::vector<int>> board = {
        {0, 0, 7, 8, 6, 0, 2, 0, 1},
        {6, 1, 0, 0, 0, 0, 0, 0, 0},
        {8, 0, 2, 4, 1, 3, 0, 0, 0},
        {2, 0, 4, 0, 0, 5, 0, 0, 9},
        {9, 8, 0, 0, 3, 0, 7, 0, 0},
        {0, 0, 3, 0, 0, 8, 0, 0, 2},
        {0, 0, 0, 3, 8, 0, 1, 9, 7},
        {0, 3, 0, 0, 0, 0, 5, 0, 4},
        {1, 9, 0, 0, 0, 0, 6, 0, 3}
    };

    ValidateSudoku sudoku(board);
    sudoku.chequeo_general();
    std::cout << "El tablero de Sudoku ingresado es válido" << std::endl;

    if (sudoku.solve_sudoku(board)) {
        sudoku.print_board(board);
    }
    else {
        std::cout << "No solution exists" << std::endl;
    }

    return 0;
}
