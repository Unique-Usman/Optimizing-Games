import chess
from stockfish import Stockfish
import chess.svg
from cairosvg import svg2png
import matplotlib.pyplot as plt
from PIL import Image
import io


fish = Stockfish(
        path="/home/malhar/.local/Programs/stockfish/src/stockfish",
        depth=1,
        parameters={"Threads": 1,
                    "Ponder": "false",
                    "Contempt": 0,
                    "Skill Level": 0,
                    }
        )


piece_symbols = {
    chess.Piece(chess.KING, chess.BLACK): '♔',
    chess.Piece(chess.QUEEN, chess.BLACK): '♕',
    chess.Piece(chess.ROOK, chess.BLACK): '♖',
    chess.Piece(chess.BISHOP, chess.BLACK): '♗',
    chess.Piece(chess.KNIGHT, chess.BLACK): '♘',
    chess.Piece(chess.PAWN, chess.BLACK): '♙',
    chess.Piece(chess.KING, chess.WHITE): '♚',
    chess.Piece(chess.QUEEN, chess.WHITE): '♛',
    chess.Piece(chess.ROOK, chess.WHITE): '♜',
    chess.Piece(chess.BISHOP, chess.WHITE): '♝',
    chess.Piece(chess.KNIGHT, chess.WHITE): '♞',
    chess.Piece(chess.PAWN, chess.WHITE): '♟',
    None: '  '
}


def display_board(board, previous_image=None):
    svg_text = chess.svg.board(board=board)
    png_output = svg2png(bytestring=svg_text)
    img_stream = io.BytesIO(png_output)
    img = Image.open(img_stream)
    if previous_image:
        previous_image.remove()
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.pause(0.5)
    plt.clf()
    return plt.imshow(img)


def print_board(board, move):
    print('\033[H\033[J', end="")
    print("      A   B   C   D   E   F   G   H")
    print("   +--------------------------------+")
    for i in range(8, 0, -1):
        print(f" {i} |", end=" ")
        for j in range(1, 9):
            square = chess.square(j - 1, i - 1)
            piece = board.piece_at(square)
            if piece is not None:
                print(f" {piece_symbols.get(piece)} ", end="")
            else:
                print("   ", end="")
            print("|", end="")
        print()
        print("   +--------------------------------+")
    print("      A   B   C   D   E   F   G   H")
    print("Previous move:", move)


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None

        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)[0]
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval = float('inf')
        best_move = None

        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)[0]
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def evaluate_board(board):
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
            ]

    knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
            ]

    bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
            ]

    rook_table = [
            0, 0, 0, 5, 5, 0, 0, 0
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0,
            ]

    queen_table = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
            ]

    king_table = [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30
            ]

    piece_tables = {
        chess.PAWN: pawn_table,
        chess.KNIGHT: knight_table,
        chess.BISHOP: bishop_table,
        chess.ROOK: rook_table,
        chess.QUEEN: queen_table,
        chess.KING: king_table
    }

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                score += piece_values[piece.piece_type] + \
                    piece_tables[piece.piece_type][square]
            else:
                score -= piece_values[piece.piece_type] + \
                    piece_tables[piece.piece_type][chess.square_mirror(square)]

    return score


def bot_move(board):
    depth = 4
    if board.fullmove_number == 1:
        return chess.Move.from_uci("e2e4")
    elif board.fullmove_number == 2:
        f6 = board.piece_at(chess.parse_square('f6'))
        d5 = board.piece_at(chess.parse_square('d5'))
        if (f6 is not None and f6.piece_type == chess.KNIGHT) or (d5 is not None and d5.piece_type == chess.PAWN):
            return chess.Move.from_uci("b1c3")
        else:
            return chess.Move.from_uci("g1f3")
    else:
        _, best_move = minimax(board, depth, float('-inf'), float('inf'), True)
        return best_move


def stockfish_move(board, depth=20):
    max_time = 50
    fish.set_fen_position(board.fen())

    best_move = fish.get_best_move_time(time=max_time)
    print(best_move)

    return best_move


def main():
    board = chess.Board()
    """
    for square in chess.SQUARES:
        print(square)
    return
    """

    disp_img = display_board(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = bot_move(board)
            board.push(move)
            # print_board(board, move)
            disp_img = display_board(board, disp_img)
        else:
            # user_move = input("Enter your move in UCI notation: ")
            user_move = stockfish_move(board)
            try:
                board.push_uci(user_move)
                # print_board(board, move)
                disp_img = display_board(board, disp_img)
            except ValueError:
                print("Invalid move, try again.")

    print("Game over. Result:", board.result())
    print(board.fullmove_number)


if __name__ == "__main__":
    main()
