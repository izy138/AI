// Global variables
let currentState = [1, 2, 3, 4, 5, 6, 7, 8, 0];
let goalState = [1, 2, 3, 4, 5, 6, 7, 8, 0];
let currentImage = null;
let moveCount = 0;
let solution = [];
let currentSolutionStep = 0;

// Initialize puzzle grid
function initializePuzzle() {
    const grid = document.getElementById('puzzleGrid');
    grid.innerHTML = '';

    for (let i = 0; i < 9; i++) {
        const tile = document.createElement('div');
        tile.className = 'puzzle-tile';
        tile.onclick = () => handleTileClick(i);
        grid.appendChild(tile);
    }

    updateDisplay();
}

// Handle image upload
document.getElementById('imageUpload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (event) {
        processImage(event.target.result);
    };
    reader.readAsDataURL(file);
});

// Process and scale image to fit puzzle grid
function processImage(imageSrc) {
    const img = new Image();
    img.onload = function () {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Set canvas size to puzzle dimensions
        canvas.width = 360; // 3x3 grid of 120px tiles
        canvas.height = 360;

        // Calculate scaling to fit image while maintaining aspect ratio
        const imgAspect = img.width / img.height;
        const canvasAspect = 1; // Square canvas

        let drawWidth, drawHeight, drawX, drawY;

        if (imgAspect > canvasAspect) {
            // Image is wider - fit to height
            drawHeight = canvas.height;
            drawWidth = drawHeight * imgAspect;
            drawX = (canvas.width - drawWidth) / 2;
            drawY = 0;
        } else {
            // Image is taller - fit to width
            drawWidth = canvas.width;
            drawHeight = drawWidth / imgAspect;
            drawX = 0;
            drawY = (canvas.height - drawHeight) / 2;
        }

        // Fill canvas with black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw scaled image
        ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);

        // Store the processed image
        currentImage = canvas.toDataURL();
        resetPuzzle();
        updateStatus('Image loaded and scaled! You can now shuffle and play.');
    };
    img.src = imageSrc;
}

// Update puzzle display
function updateDisplay() {
    const tiles = document.querySelectorAll('.puzzle-tile');

    tiles.forEach((tile, index) => {
        const value = currentState[index];
        tile.innerHTML = '';

        if (value === 0) {
            // Empty tile - clear all styles and show as empty
            tile.className = 'puzzle-tile empty';
            tile.style.backgroundImage = 'none';
            tile.style.background = '#222';
            tile.style.display = 'block';
        } else {
            tile.className = 'puzzle-tile';

            if (currentImage) {
                // Calculate the position of this tile piece in the original image
                const row = Math.floor((value - 1) / 3);
                const col = (value - 1) % 3;

                tile.style.backgroundImage = `url(${currentImage})`;
                tile.style.backgroundPosition = `${-col * 120}px ${-row * 120}px`;
                tile.style.backgroundSize = '360px 360px'; // Full puzzle size
                tile.style.backgroundRepeat = 'no-repeat';
            } else {
                // Default colored tiles with numbers
                tile.style.backgroundImage = 'none';
                tile.style.background = `hsl(${value * 40}, 70%, 60%)`;
                tile.style.display = 'flex';
                tile.style.alignItems = 'center';
                tile.style.justifyContent = 'center';
                tile.style.fontSize = '24px';
                tile.style.fontWeight = 'bold';
                tile.style.color = 'white';
                tile.textContent = value;
            }

            // Add tile number overlay (only for non-empty tiles)
            const numberOverlay = document.createElement('div');
            numberOverlay.className = 'tile-number';
            numberOverlay.textContent = value;
            tile.appendChild(numberOverlay);
        }
    });

    document.getElementById('moveCount').textContent = moveCount;

    if (isPuzzleSolved()) {
        showSuccess();
    }
}

// Handle tile clicks
function handleTileClick(index) {
    const emptyIndex = currentState.indexOf(0);

    if (isValidMove(index, emptyIndex)) {
        // Swap tiles
        [currentState[index], currentState[emptyIndex]] = [currentState[emptyIndex], currentState[index]];
        moveCount++;
        updateDisplay();
    }
}

// Check if move is valid
function isValidMove(tileIndex, emptyIndex) {
    const tileRow = Math.floor(tileIndex / 3);
    const tileCol = tileIndex % 3;
    const emptyRow = Math.floor(emptyIndex / 3);
    const emptyCol = emptyIndex % 3;

    return Math.abs(tileRow - emptyRow) + Math.abs(tileCol - emptyCol) === 1;
}

// Shuffle puzzle
function shufflePuzzle() {
    // Start from solved state and make random valid moves
    currentState = [...goalState];

    for (let i = 0; i < 100; i++) {
        const emptyIndex = currentState.indexOf(0);
        const validMoves = [];

        // Find all valid moves
        for (let j = 0; j < 9; j++) {
            if (isValidMove(j, emptyIndex)) {
                validMoves.push(j);
            }
        }

        // Make a random valid move
        if (validMoves.length > 0) {
            const randomMove = validMoves[Math.floor(Math.random() * validMoves.length)];
            [currentState[randomMove], currentState[emptyIndex]] = [currentState[emptyIndex], currentState[randomMove]];
        }
    }

    moveCount = 0;
    solution = [];
    currentSolutionStep = 0;
    updateDisplay();
    updateStatus('Puzzle shuffled! Start solving or click "Find Solution".');
    document.getElementById('solutionSteps').innerHTML = '<p>Click "Find Solution" to see the optimal path.</p>';
}

// A* Algorithm Implementation
class PuzzleNode {
    constructor(state, parent = null, move = null, depth = 0) {
        this.state = [...state];
        this.parent = parent;
        this.move = move;
        this.depth = depth;
        this.heuristic = this.calculateManhattanDistance();
        this.f = this.depth + this.heuristic;
    }

    calculateManhattanDistance() {
        let distance = 0;
        for (let i = 0; i < 9; i++) {
            if (this.state[i] !== 0) {
                const currentRow = Math.floor(i / 3);
                const currentCol = i % 3;
                const targetPos = this.state[i] - 1;
                const targetRow = Math.floor(targetPos / 3);
                const targetCol = targetPos % 3;
                distance += Math.abs(currentRow - targetRow) + Math.abs(currentCol - targetCol);
            }
        }
        return distance;
    }

    getStateString() {
        return this.state.join(',');
    }

    getNeighbors() {
        const neighbors = [];
        const emptyIndex = this.state.indexOf(0);
        const emptyRow = Math.floor(emptyIndex / 3);
        const emptyCol = emptyIndex % 3;

        const directions = [
            { row: -1, col: 0, name: 'UP' },
            { row: 1, col: 0, name: 'DOWN' },
            { row: 0, col: -1, name: 'LEFT' },
            { row: 0, col: 1, name: 'RIGHT' }
        ];

        for (const dir of directions) {
            const newRow = emptyRow + dir.row;
            const newCol = emptyCol + dir.col;

            if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
                const newIndex = newRow * 3 + newCol;
                const newState = [...this.state];
                [newState[emptyIndex], newState[newIndex]] = [newState[newIndex], newState[emptyIndex]];

                neighbors.push(new PuzzleNode(newState, this, dir.name, this.depth + 1));
            }
        }

        return neighbors;
    }
}

// A* Search Implementation
function aStarSearch(start, goal) {
    const startNode = new PuzzleNode(start);
    const goalString = goal.join(',');

    if (startNode.getStateString() === goalString) {
        return { path: [], nodesExplored: 0, algorithm: 'A*' };
    }

    const openSet = [startNode];
    const closedSet = new Set();
    const gScores = new Map();
    gScores.set(startNode.getStateString(), 0);

    let nodesExplored = 0;

    while (openSet.length > 0) {
        // Find node with lowest f score
        openSet.sort((a, b) => a.f - b.f);
        const currentNode = openSet.shift();
        const currentString = currentNode.getStateString();

        if (currentString === goalString) {
            // Reconstruct path
            const path = [];
            let node = currentNode;
            while (node.parent) {
                path.unshift({ move: node.move, state: [...node.state] });
                node = node.parent;
            }
            return { path, nodesExplored, algorithm: 'A*' };
        }

        closedSet.add(currentString);
        nodesExplored++;

        // Explore neighbors
        for (const neighbor of currentNode.getNeighbors()) {
            const neighborString = neighbor.getStateString();

            if (closedSet.has(neighborString)) {
                continue;
            }

            const tentativeGScore = currentNode.depth + 1;
            const existingGScore = gScores.get(neighborString);

            if (existingGScore === undefined || tentativeGScore < existingGScore) {
                gScores.set(neighborString, tentativeGScore);

                // Remove existing node with same state from openSet
                const existingIndex = openSet.findIndex(node => node.getStateString() === neighborString);
                if (existingIndex !== -1) {
                    openSet.splice(existingIndex, 1);
                }

                openSet.push(neighbor);
            }
        }

        // Prevent infinite loops in case of complex puzzles
        if (nodesExplored > 100000) {
            break;
        }
    }

    return { path: [], nodesExplored, algorithm: 'A*' };
}

// BFS Search Implementation
function bfsSearch(start, goal) {
    const startString = start.join(',');
    const goalString = goal.join(',');

    if (startString === goalString) {
        return { path: [], nodesExplored: 0, algorithm: 'BFS' };
    }

    const queue = [{ state: [...start], path: [] }];
    const visited = new Set([startString]);
    let nodesExplored = 0;

    while (queue.length > 0) {
        const current = queue.shift();
        nodesExplored++;

        // Get neighbors using our existing PuzzleNode class
        const tempNode = new PuzzleNode(current.state);
        const neighbors = tempNode.getNeighbors();

        for (const neighbor of neighbors) {
            const neighborString = neighbor.getStateString();

            if (neighborString === goalString) {
                // Found solution - build path
                const solutionPath = [...current.path, { move: neighbor.move, state: [...neighbor.state] }];
                return { path: solutionPath, nodesExplored, algorithm: 'BFS' };
            }

            if (!visited.has(neighborString)) {
                visited.add(neighborString);
                queue.push({
                    state: [...neighbor.state],
                    path: [...current.path, { move: neighbor.move, state: [...neighbor.state] }]
                });
            }
        }

        // Prevent browser freeze
        if (nodesExplored > 100000) {
            return { path: [], nodesExplored, algorithm: 'BFS', timeout: true };
        }
    }

    return { path: [], nodesExplored, algorithm: 'BFS' };
}
function dfsSearch(start, goal, maxDepth = 75) {
    console.log('=== DFS DEBUG START ===');
    console.log('Start state:', start);
    console.log('Goal state:', goal);
    console.log('Max depth:', maxDepth);

    const startString = start.join(',');
    const goalString = goal.join(',');

    if (startString === goalString) {
        console.log('Already at goal!');
        return { path: [], nodesExplored: 0, algorithm: 'DFS' };
    }

    const stack = [{ state: [...start], path: [], depth: 0 }];
    const visited = new Set();
    let nodesExplored = 0;
    let maxDepthReached = 0;
    let solutionsAttempted = 0;

    console.log('Starting DFS search...');

    while (stack.length > 0) {
        const current = stack.pop();
        const currentString = current.state.join(',');
        nodesExplored++;

        // Track max depth reached
        if (current.depth > maxDepthReached) {
            maxDepthReached = current.depth;
            console.log(`New max depth reached: ${maxDepthReached}`);
        }

        // Log every 1000 nodes
        if (nodesExplored % 1000 === 0) {
            console.log(`Nodes explored: ${nodesExplored}, Current depth: ${current.depth}, Stack size: ${stack.length}`);
        }

        // Check if we found the goal
        if (currentString === goalString) {
            console.log('=== SOLUTION FOUND! ===');
            console.log(`Path length: ${current.path.length}`);
            console.log(`Nodes explored: ${nodesExplored}`);
            console.log(`Max depth reached: ${maxDepthReached}`);
            return { path: current.path, nodesExplored, algorithm: 'DFS' };
        }

        // Skip if too deep
        if (current.depth >= maxDepth) {
            continue;
        }

        // Skip if already visited
        if (visited.has(currentString)) {
            continue;
        }

        visited.add(currentString);

        // Get neighbors
        const tempNode = new PuzzleNode(current.state);
        const neighbors = tempNode.getNeighbors();

        console.log(`Depth ${current.depth}: Found ${neighbors.length} neighbors`);

        // Add neighbors to stack (in reverse order so DFS explores first neighbor first)
        for (let i = neighbors.length - 1; i >= 0; i--) {
            const neighbor = neighbors[i];
            const neighborString = neighbor.getStateString();

            if (!visited.has(neighborString)) {
                stack.push({
                    state: [...neighbor.state],
                    path: [...current.path, { move: neighbor.move, state: [...neighbor.state] }],
                    depth: current.depth + 1
                });
            }
        }

        // Safety check to prevent infinite loops
        if (nodesExplored > 200000) {
            console.log('=== SEARCH TERMINATED (too many nodes) ===');
            console.log(`Nodes explored: ${nodesExplored}`);
            console.log(`Max depth reached: ${maxDepthReached}`);
            console.log(`Stack size when terminated: ${stack.length}`);
            break;
        }
    }

    console.log('=== DFS FAILED ===');
    console.log(`Total nodes explored: ${nodesExplored}`);
    console.log(`Max depth reached: ${maxDepthReached}`);
    console.log(`Final stack size: ${stack.length}`);
    console.log('=== DFS DEBUG END ===');

    return { path: [], nodesExplored, algorithm: 'DFS', depthLimited: maxDepth };
}
// Solve puzzle function
async function solvePuzzle() {
    if (isPuzzleSolved()) {
        updateStatus('Puzzle is already solved!');
        return;
    }

    const selectedAlgorithm = document.getElementById('algorithmSelect').value;
    updateStatus(`Solving puzzle using ${selectedAlgorithm.toUpperCase()}...`);

    setTimeout(() => {
        const startTime = performance.now();
        let result;

        switch (selectedAlgorithm) {
            case 'astar':
                result = aStarSearch([...currentState], goalState);
                break;
            case 'bfs':
                result = bfsSearch([...currentState], goalState);
                break;
            case 'dfs':
                result = dfsSearch([...currentState], goalState);
                break;
        }

        const endTime = performance.now();
        const solveTime = endTime - startTime;

        if (result.path.length > 0) {
            displaySolution(result, solveTime);
            updateStatus(`${result.algorithm} found solution!`);

            // Automatically start animation
            setTimeout(() => {
                animateSolution(result.path);
            }, 1000);

        } else {
            updateStatus(`${result.algorithm} could not find solution`);
        }
    }, 100);
}

// Animation function
function animateSolution(solutionPath) {
    if (!solutionPath || solutionPath.length === 0) return;

    let stepIndex = 0;
    const speed = 500; // 500ms between moves

    function nextStep() {
        if (stepIndex >= solutionPath.length) {
            updateStatus('Solution animation completed!');
            return;
        }

        // Update the current state to the next step
        currentState = [...solutionPath[stepIndex].state];
        updateDisplay();

        // Highlight current step in solution display
        document.querySelectorAll('.solution-step').forEach((step, index) => {
            step.style.background = index === stepIndex ? '#e3f2fd' : 'white';
        });

        updateStatus(`Step ${stepIndex + 1} of ${solutionPath.length}: ${solutionPath[stepIndex].move}`);
        stepIndex++;

        setTimeout(nextStep, speed);
    }

    nextStep();
}

// Display solution steps
function displaySolution(result, solveTime) {
    const container = document.getElementById('solutionSteps');
    let html = `
        <div style="background: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; margin-top: 10px; border-left: 4px solid #4CAF50;">
            <strong>${result.algorithm} Solution Found!</strong><br>
            Steps: ${result.path.length}<br>
            Nodes explored: ${result.nodesExplored}<br>
            Time: ${Math.round(solveTime)}ms
            ${result.timeout ? '<br><span style="color: #FF9800;">⚠️ Search limited to prevent timeout</span>' : ''}
            ${result.depthLimited ? `<br><span style="color: #FF9800;">⚠️ Depth limited to ${result.depthLimited}</span>` : ''}
        </div>
    `;

    result.path.forEach((step, index) => {
        html += `
            <div class="solution-step" id="step-${index}">
                <strong>Step ${index + 1}:</strong> Move ${step.move}
                <div style="font-size: 12px; margin-top: 5px; font-family: monospace;">
                    ${formatStateForDisplay(step.state)}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Compare all algorithms function
async function compareAllAlgorithms() {
    if (isPuzzleSolved()) {
        updateStatus('Please shuffle the puzzle first before comparing algorithms!');
        return;
    }

    updateStatus('Running all algorithms for comparison...');
    document.getElementById('solutionSteps').innerHTML = '<div class="loading">Comparing A*, BFS, and DFS algorithms</div>';

    const testState = [...currentState];
    const results = [];

    try {
        // Test A* Algorithm
        console.log('Testing A*...');
        const startTime1 = performance.now();
        const astarResult = aStarSearch(testState, goalState);
        const endTime1 = performance.now();
        astarResult.time = Math.round(endTime1 - startTime1);
        results.push(astarResult);
        console.log('A* completed:', astarResult);

        // Test BFS Algorithm  
        console.log('Testing BFS...');
        const startTime2 = performance.now();
        const bfsResult = bfsSearch(testState, goalState);
        const endTime2 = performance.now();
        bfsResult.time = Math.round(endTime2 - startTime2);
        results.push(bfsResult);
        console.log('BFS completed:', bfsResult);

        // Test DFS Algorithm
        console.log('Testing DFS...');
        const startTime3 = performance.now();
        const dfsResult = dfsSearch(testState, goalState, 50);
        const endTime3 = performance.now();
        dfsResult.time = Math.round(endTime3 - startTime3);
        results.push(dfsResult);
        console.log('DFS completed:', dfsResult);

        // Display results
        displayComparisonResults(results);
        updateStatus('Algorithm comparison completed!');

        // Set best solution
        const bestResult = results.find(r => r.algorithm === 'A*' && r.path.length > 0) ||
            results.find(r => r.path.length > 0);
        if (bestResult && bestResult.path.length > 0) {
            solution = bestResult.path;
            currentSolutionStep = 0;
            document.getElementById('nextStepBtn').disabled = false;
        }

    } catch (error) {
        console.error('Error in comparison:', error);
        updateStatus('Error during algorithm comparison');
    }
}

// Display comparison results
function displayComparisonResults(results) {
    const comparisonDiv = document.getElementById('comparisonResults');
    const tableBody = document.getElementById('comparisonTableBody');

    // Find best results for highlighting
    const validResults = results.filter(r => r.path.length > 0);
    let bestSteps = Math.min(...validResults.map(r => r.path.length));
    let bestNodes = Math.min(...validResults.map(r => r.nodesExplored));
    let bestTime = Math.min(...validResults.map(r => r.time));

    let tableHTML = '';

    results.forEach(result => {
        const hasTimeout = result.timeout ? ' (Timeout)' : '';
        const hasDepthLimit = result.depthLimited ? ` (Depth ${result.depthLimited})` : '';
        let status;
        if (result.path.length > 0) {
            status = 'Success' + hasTimeout + hasDepthLimit;
        } else if (result.algorithm === 'DFS') {
            status = 'Failed (DFS limitation)';
        } else {
            status = 'Failed' + hasTimeout + hasDepthLimit;
        }

        const stepsClass = result.path.length === bestSteps && result.path.length > 0 ? 'best-result' : '';
        const nodesClass = result.nodesExplored === bestNodes && result.path.length > 0 ? 'best-result' : '';
        const timeClass = result.time === bestTime && result.path.length > 0 ? 'best-result' : '';

        tableHTML += `
            <tr>
                <td><strong>${result.algorithm}</strong></td>
                <td class="${stepsClass}">${result.path.length > 0 ? result.path.length : 'N/A'}</td>
                <td class="${nodesClass}">${result.nodesExplored.toLocaleString()}</td>
                <td class="${timeClass}">${result.time}</td>
                <td>${status}</td>
            </tr>
        `;
    });

    tableBody.innerHTML = tableHTML;
    comparisonDiv.style.display = 'block';

    // Also update the solution display
    let solutionHTML = '<h3>Comparison Complete</h3>';
    solutionHTML += '<p>See the comparison table below for detailed results.</p>';

    // Show the best solution steps
    const bestResult = validResults.find(r => r.algorithm === 'A*') || validResults[0];
    if (bestResult) {
        solutionHTML += `<h4>Best Solution (${bestResult.algorithm}):</h4>`;
        bestResult.path.forEach((step, index) => {
            solutionHTML += `
                <div class="solution-step" id="step-${index}">
                    <strong>Step ${index + 1}:</strong> Move ${step.move}
                    <div style="font-size: 12px; margin-top: 5px; font-family: monospace;">
                        ${formatStateForDisplay(step.state)}
                    </div>
                </div>
            `;
        });
    }

    document.getElementById('solutionSteps').innerHTML = solutionHTML;
}

// Format state for display
function formatStateForDisplay(state) {
    let result = '';
    for (let i = 0; i < 9; i++) {
        if (i % 3 === 0 && i > 0) result += '<br>';
        result += (state[i] === 0 ? '_' : state[i]) + ' ';
    }
    return result;
}

// Play next solution step
function playNextStep() {
    if (currentSolutionStep < solution.length) {
        currentState = [...solution[currentSolutionStep].state];
        updateDisplay();

        // Highlight current step
        document.querySelectorAll('.solution-step').forEach((step, index) => {
            step.style.background = index === currentSolutionStep ? '#e3f2fd' : 'white';
        });

        currentSolutionStep++;

        if (currentSolutionStep >= solution.length) {
            document.getElementById('nextStepBtn').disabled = true;
            updateStatus('Solution completed!');
        }
    }
}

// Check if puzzle is solved
function isPuzzleSolved() {
    return currentState.every((value, index) => value === goalState[index]);
}

// Reset puzzle
function resetPuzzle() {
    currentState = [...goalState];
    moveCount = 0;
    solution = [];
    currentSolutionStep = 0;
    updateDisplay();
    updateStatus('Puzzle reset to solved state.');
    document.getElementById('nextStepBtn').disabled = true;
    document.getElementById('solutionSteps').innerHTML = '<p>Upload an image and click "Find Solution" to see the optimal path.</p>';
}

// Update status
function updateStatus(message) {
    document.getElementById('gameStatus').textContent = message;
}

// Show success message
function showSuccess() {
    const existingMessage = document.querySelector('.success-message');
    if (existingMessage) return;

    const message = document.createElement('div');
    message.className = 'success-message';
    message.textContent = `Congratulations! Puzzle solved in ${moveCount} moves!`;

    const container = document.querySelector('.left-panel');
    container.insertBefore(message, container.firstChild);

    setTimeout(() => {
        if (message.parentNode) {
            message.parentNode.removeChild(message);
        }
    }, 5000);

    updateStatus('Puzzle solved!');
}

// Update algorithm info when selection changes
document.getElementById('algorithmSelect').addEventListener('change', function () {
    const algorithm = this.value;
    const titleElement = document.getElementById('algorithmTitle');
    const descElement = document.getElementById('algorithmDescription');
    const currentAlgElement = document.getElementById('currentAlgorithm');
    const heuristicElement = document.getElementById('heuristicDisplay');

    switch (algorithm) {
        case 'astar':
            titleElement.textContent = 'Algorithm: A* with Manhattan Distance Heuristic';
            descElement.textContent = 'A* uses Manhattan distance heuristic to find optimal solutions efficiently. It combines the actual cost to reach a state with an estimated cost to the goal, guaranteeing the shortest path while exploring fewer nodes than uninformed search.';
            currentAlgElement.textContent = 'A* Search';
            heuristicElement.textContent = 'Manhattan Distance';
            break;
        case 'bfs':
            titleElement.textContent = 'Algorithm: Breadth-First Search (BFS)';
            descElement.textContent = 'BFS explores all nodes at the current depth before moving to nodes at the next depth. It guarantees finding the optimal solution but may explore many more nodes than A*. It uses no heuristic information.';
            currentAlgElement.textContent = 'BFS';
            heuristicElement.textContent = 'None (Uninformed)';
            break;
        case 'dfs':
            titleElement.textContent = 'Algorithm: Depth-First Search (DFS) - Limited Depth';
            descElement.textContent = 'DFS explores as far as possible along each branch before backtracking. For 8-puzzle, we limit the depth to prevent infinite loops. DFS does not guarantee optimal solutions and may find longer paths.';
            currentAlgElement.textContent = 'DFS (Limited)';
            heuristicElement.textContent = 'None (Uninformed)';
            break;
    }
});

// Initialize the game
initializePuzzle();
updateStatus('Upload an image to start playing!');