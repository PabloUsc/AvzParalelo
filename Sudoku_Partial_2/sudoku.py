'''
board = [
    ["5","3",".",".","7",".",".",".","."]
    ,["6",".",".","1","9","5",".",".","."]
    ,[".","9","8",".",".",".",".","6","."]
    ,["8",".",".",".","6",".",".",".","3"]
    ,["4",".",".","8",".","3",".",".","1"]
    ,["7",".",".",".","2",".",".",".","6"]
    ,[".","6",".",".",".",".","2","8","."]
    ,[".",".",".","4","1","9",".",".","5"]
    ,[".",".",".",".","8",".",".","7","9"]
]
'''

board = [
    [ "." , "." , "7" , "8" , "6" , "." , "2" , "." , "1" ],
    [ "6" , "1" , "." , "." , "." , "." , "." , "." , "." ],
    [ "8" , "." , "2" , "4" , "1" , "3" , "." , "." , "." ],
    [ "2" , "." , "4" , "." , "." , "5" , "." , "." , "9" ],
    [ "9" , "8" , "." , "." , "3" , "." , "7" , "." , "." ],
    [ "." , "." , "3" , "." , "." , "8" , "." , "." , "2" ],
    [ "." , "." , "." , "3" , "8" , "." , "1" , "9" , "7" ],
    [ "." , "3" , "." , "." , "." , "." , "5" , "." , "4" ],
    [ "1" , "9" , "." , "." , "." , "." , "6" , "." , "3" ]
    ]

class ValidateSudoku:
    def __init__(self,tablero) -> None:
        self.tablero = tablero
        self.lista_invertida = list()

    def chequeo_general(self):
        """
        Chequear que el tablero introducido sea un tablero 9x9
        """
        #assert
        assert len(self.tablero) == 9, "El tablero ingresado no respeta el formato 9x9" #filas
        for fila in self.tablero:
            assert len(fila) == 9, "El tablero ingresado no respeta el formato 9x9"

   
    def chequeo_filas(self,lista_a_chequear='tablero_general'):
        if lista_a_chequear == 'tablero_general':
            lista_a_chequear = self.tablero

        for fila in lista_a_chequear:
            for elemento in fila:
                if elemento != '.':
                    assert fila.count(elemento) == 1, "El tablero ingresado no es válido"


    def chequeo_columnas(self):

        for column_index in range(0,9):
            for row_index in range(0,9):
                self.lista_invertida.append(self.tablero[row_index][column_index])
           
            self.chequeo_filas([self.lista_invertida])

            self.lista_invertida.clear()


    def chequeo_subcuadros(self):
        #funcion mayor
        #tenemos 9 subcuadros = chequear de 3 en 3
        # de mis primeras 3 filas -> subcuadros del 0 al 3, 3 al 6, 6 al 9
        self.chequeo_3_subcuadros(0,3)
        self.chequeo_3_subcuadros(3,6)
        self.chequeo_3_subcuadros(6,9)

    def chequeo_3_subcuadros(self,rango1,rango2):
        self.lista_invertida.clear()
        for row_index in range(0,9):
            if row_index == 3 or row_index == 6:
                self.lista_invertida.clear()
            for column_index in range(rango1,rango2):
                self.lista_invertida.append(self.tablero[column_index][row_index])
                if len(self.lista_invertida) == 9:
                    self.chequeo_filas([self.lista_invertida])
                #chequeo filas
       

def encuentra_vacio(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == '.':
                return row, col
    return -1, -1  #no encuentra vacío


def posible(board, fila, col, num):    
    if num in board[fila]:
        return False
    if num in [board[i][col] for i in range(9)]:
        return False
    fila_inicio, col_inicio = 3 * (fila // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[i + fila_inicio][j + col_inicio] == num:
                return False
    return True


def solucionar_sudoku(board):
    row, col = encuentra_vacio(board)

    # Resuleto de no encontrar
    if row == -1 and col == -1:
        return True

    #Prueba de números (bruteforce)
    for num in map(str, range(1, 10)):
        if posible(board, row, col, num):
            board[row][col] = num

            # Recursividad
            if solucionar_sudoku(board):
                return True
            board[row][col] = '.'
    return False


def print_board(board):
    for row in board:
        print(" ".join(row))

#instanciar objeto
if __name__ == "__main__":
    sudoku = ValidateSudoku(board)
    sudoku.chequeo_general()
    sudoku.chequeo_filas()
    sudoku.chequeo_columnas()
    sudoku.chequeo_subcuadros()
    if solucionar_sudoku(board):
        print("La solución es:")
        print_board(board)
    else:
        print("No existe solución")
