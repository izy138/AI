let board = Array(9).fill(null);
let currentPlayer = 'X';
let gameActive = true;
let isPaused = false;
let moveHistory = [];
let aiTimeout = null;

const winningLines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
];

function initBoard() {
    const boardElement = document.getElementById('board');
    boardElement.innerHTML = '';
    for (let i = 0; i < 9; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.index = i;
        cell.addEventListener('click', handleCellClick);
        boardElement.appendChild(cell);
    }
}

function calculateWinner(squares) {
    for (let line of winningLines) {
        const [a, b, c] = line;
        if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
            return { winner: squares[a], line };
        }
    }
    return null;
}

function minimax(squares, depth, isMaximizing, alpha = -Infinity, beta = Infinity, useAlphaBeta = false) {
    let nodesExplored = 1;
    let nodesPruned = 0;

    const result = calculateWinner(squares);
    if (result) {
        const score = result.winner === 'O' ? 10 - depth : depth - 10;
        return { score, nodes: nodesExplored, pruned: nodesPruned };
    }
    if (!squares.includes(null)) {
        return { score: 0, nodes: nodesExplored, pruned: nodesPruned };
    }

    if (isMaximizing) {
        let maxScore = -Infinity;
        for (let i = 0; i < 9; i++) {
            if (squares[i] === null) {
                squares[i] = 'O';
                const result = minimax(squares, depth + 1, false, alpha, beta, useAlphaBeta);
                squares[i] = null;
                maxScore = Math.max(maxScore, result.score);
                nodesExplored += result.nodes;
                nodesPruned += result.pruned;

                if (useAlphaBeta) {
                    alpha = Math.max(alpha, maxScore);
                    if (beta <= alpha) {
                        nodesPruned++;
                        break;
                    }
                }
            }
        }
        return { score: maxScore, nodes: nodesExplored, pruned: nodesPruned };
    } else {
        let minScore = Infinity;
        for (let i = 0; i < 9; i++) {
            if (squares[i] === null) {
                squares[i] = 'X';
                const result = minimax(squares, depth + 1, true, alpha, beta, useAlphaBeta);
                squares[i] = null;
                minScore = Math.min(minScore, result.score);
                nodesExplored += result.nodes;
                nodesPruned += result.pruned;

                if (useAlphaBeta) {
                    beta = Math.min(beta, minScore);
                    if (beta <= alpha) {
                        nodesPruned++;
                        break;
                    }
                }
            }
        }
        return { score: minScore, nodes: nodesExplored, pruned: nodesPruned };
    }
}

function getBestMove(squares, player, algorithm) {
    const startTime = performance.now();
    let bestMove = -1;
    let bestScore = player === 'O' ? -Infinity : Infinity;
    let totalNodes = 0;
    let totalPruned = 0;
    const useAlphaBeta = algorithm === 'alphabeta';

    for (let i = 0; i < 9; i++) {
        if (squares[i] === null) {
            squares[i] = player;
            const result = minimax(squares, 0, player === 'X', -Infinity, Infinity, useAlphaBeta);
            squares[i] = null;
            totalNodes += result.nodes;
            totalPruned += result.pruned;

            if ((player === 'O' && result.score > bestScore) ||
                (player === 'X' && result.score < bestScore)) {
                bestScore = result.score;
                bestMove = i;
            }
        }
    }

    const endTime = performance.now();
    const decisionTime = (endTime - startTime).toFixed(2);

    return {
        move: bestMove,
        time: decisionTime,
        nodes: totalNodes,
        pruned: totalPruned
    };
}

function handleCellClick(event) {
    const index = parseInt(event.target.dataset.index);
    if (board[index] || !gameActive || isPaused) return;

    const mode = document.getElementById('gameMode').value;
    if (mode === 'ai-ai') return;

    if (mode === 'human-ai') {
        const humanFirst = document.getElementById('playOrder').value === 'first';
        const humanSymbol = humanFirst ? 'X' : 'O';
        if (currentPlayer !== humanSymbol) return;
    }

    makeMove(index);
}

function makeMove(index) {
    board[index] = currentPlayer;
    const cell = document.querySelector(`[data-index="${index}"]`);
    cell.textContent = currentPlayer;
    cell.classList.add('filled', currentPlayer.toLowerCase());

    moveHistory.push({ player: currentPlayer, position: index });
    updateHistoryDisplay();

    const result = calculateWinner(board);
    if (result) {
        gameActive = false;
        document.getElementById('status').textContent = `${result.winner} Wins!`;
        result.line.forEach(i => {
            document.querySelector(`[data-index="${i}"]`).classList.add('winning');
        });
        return;
    }

    if (!board.includes(null)) {
        gameActive = false;
        document.getElementById('status').textContent = "It's a Draw!";
        return;
    }

    currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
    document.getElementById('status').textContent = `Current Turn: ${currentPlayer}`;

    const mode = document.getElementById('gameMode').value;
    if (mode === 'human-ai') {
        const humanFirst = document.getElementById('playOrder').value === 'first';
        const humanSymbol = humanFirst ? 'X' : 'O';
        if (currentPlayer !== humanSymbol) {
            aiTimeout = setTimeout(aiMove, 300);
        }
    } else if (mode === 'ai-ai') {
        const speed = parseInt(document.getElementById('speed').value);
        aiTimeout = setTimeout(aiMove, speed);
    }
}

function aiMove() {
    if (!gameActive || isPaused) return;

    const mode = document.getElementById('gameMode').value;
    let algorithm;

    if (mode === 'human-ai') {
        algorithm = document.getElementById('algorithm').value;
    } else {
        algorithm = currentPlayer === 'X' ?
            document.getElementById('algorithmX').value :
            document.getElementById('algorithmO').value;
    }

    const result = getBestMove([...board], currentPlayer, algorithm);
    updateMetrics(result.time, result.nodes, result.pruned);

    if (result.move !== -1) {
        makeMove(result.move);
    }
}

function updateMetrics(time, nodes, pruned) {
    document.getElementById('timeMetric').textContent = `${time} ms`;
    document.getElementById('nodesMetric').textContent = nodes;
    const efficiency = nodes > 0 ? ((pruned / nodes) * 100).toFixed(1) : 0;
    document.getElementById('pruningMetric').textContent = `${efficiency}%`;
}

function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    if (moveHistory.length === 0) {
        historyList.innerHTML = '<div style="color: #9ca3af; font-style: italic;">No moves yet</div>';
        return;
    }

    historyList.innerHTML = moveHistory
        .map((move, index) => `
            <div class="history-item">
                <strong>${index + 1}.</strong> Player ${move.player} → Cell ${move.position + 1}
            </div>
        `)
        .join('');
}

function onModeChange() {
    const mode = document.getElementById('gameMode').value;
    const humanAiSettings = document.getElementById('humanAiSettings');
    const aiAiSettings = document.getElementById('aiAiSettings');
    const pauseBtn = document.getElementById('pauseBtn');

    humanAiSettings.style.display = mode === 'human-ai' ? 'block' : 'none';
    aiAiSettings.style.display = mode === 'ai-ai' ? 'block' : 'none';
    pauseBtn.style.display = mode === 'ai-ai' ? 'flex' : 'none';

    resetGame();
}

function togglePause() {
    isPaused = !isPaused;
    const pauseBtn = document.getElementById('pauseBtn');
    pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';

    if (!isPaused) {
        const mode = document.getElementById('gameMode').value;
        if (mode === 'ai-ai' && gameActive) {
            aiMove();
        }
    }
}

function updateSpeedLabel() {
    const speed = document.getElementById('speed').value;
    document.getElementById('speedValue').textContent = `${speed} ms`;
}

function resetGame() {
    if (aiTimeout) clearTimeout(aiTimeout);

    board = Array(9).fill(null);
    currentPlayer = 'X';
    gameActive = true;
    isPaused = false;
    moveHistory = [];

    initBoard();

    document.getElementById('status').textContent = 'Current Turn: X';
    document.getElementById('pauseBtn').textContent = 'Pause';
    updateMetrics(0, 0, 0);
    updateHistoryDisplay();

    const mode = document.getElementById('gameMode').value;
    if (mode === 'human-ai') {
        const humanFirst = document.getElementById('playOrder').value === 'first';
        if (!humanFirst) {
            setTimeout(aiMove, 500);
        }
    } else if (mode === 'ai-ai') {
        setTimeout(aiMove, 500);
    }
}

// Event listeners
document.getElementById('gameMode').addEventListener('change', onModeChange);
document.getElementById('algorithm').addEventListener('change', resetGame);
document.getElementById('playOrder').addEventListener('change', resetGame);
document.getElementById('algorithmX').addEventListener('change', resetGame);
document.getElementById('algorithmO').addEventListener('change', resetGame);
document.getElementById('speed').addEventListener('input', updateSpeedLabel);
document.getElementById('resetBtn').addEventListener('click', resetGame);
document.getElementById('pauseBtn').addEventListener('click', togglePause);

// Initialize the game
initBoard();