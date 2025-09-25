// Global variables
let currentState = [1, 2, 3, 4, 5, 6, 7, 8, 0];
let goalState = [1, 2, 3, 4, 5, 6, 7, 8, 0];
let currentImage = null;
let moveCount = 0;
let solution = [];
let currentSolutionStep = 0;
let savedShuffledState = null;
let comparisonResults = null;
let manualPlayStartTime = null;
let manualPlayTimer = null;
let gameTimer = 0;

// Track if manual play is active (before solve button is clicked)
let manualPlayActive = true;


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

    updateDisplay(false);
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
                tile.style.background = '#adb0a9ff';//`hsl(${value * 40}, 70%, 60%)`;
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

    const moveCountElement = document.getElementById('moveCount');
    if (moveCountElement) moveCountElement.textContent = moveCount;

    // if (checkSuccess && isPuzzleSolved() && moveCount > 0) { // Only show success if moves were made
    //     showSuccess();
    // }

    if (isPuzzleSolved() && moveCount > 0) {
        showSuccess();
    }
}

// Handle tile clicks
function handleTileClick(index) {
    // Start timer on first move
    if (!manualPlayStartTime && moveCount === 0) {
        manualPlayStartTime = Date.now();
        gameTimer = 0;
        startGameTimer();
    }

    const emptyIndex = currentState.indexOf(0);

    if (isValidMove(index, emptyIndex)) {
        // Swap tiles
        [currentState[index], currentState[emptyIndex]] = [currentState[emptyIndex], currentState[index]];
        moveCount++;
        updateDisplay();
        // Update manual step counter if manual play is active
        if (manualPlayActive) {
            const stepCounter = document.getElementById('stepCounter');
            if (stepCounter) {
                stepCounter.textContent = `Step ${moveCount}`;
            }
        }
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
    savedShuffledState = [...currentState]; // Save the shuffled state
    manualPlayStartTime = null;
    gameTimer = 0;
    updateTimerDisplay();
    stopGameTimer();

    // Enable manual play tracking after shuffle
    manualPlayActive = true;

    updateDisplay();

    updateStatus('Puzzle shuffled! Start solving or click "Find Solution".');
    // document.getElementById('solutionSteps').innerHTML = '<p>Click "Find Solution" to see the optimal path.</p>';


    const container = document.getElementById('solutionSteps');
    let html = `Click Shuffle Puzzle, and then Solve Puzzle to see the Algorithm Comparisons.`;
    container.innerHTML = html;
    const stepCounter = document.getElementById('stepCounter');
    if (stepCounter) stepCounter.textContent = 'Step 0';

    const comparisonResults = document.getElementById('comparisonResults');
    comparisonResults.style.display = 'none';

}

function startGameTimer() {
    // Clear any existing timer
    if (manualPlayTimer) {
        clearInterval(manualPlayTimer);
    }

    manualPlayTimer = setInterval(() => {
        if (manualPlayStartTime) {
            gameTimer = Math.floor((Date.now() - manualPlayStartTime) / 1000);
            updateTimerDisplay();
        }
    }, 1000);
}

function stopGameTimer() {
    if (manualPlayTimer) {
        clearInterval(manualPlayTimer);
        manualPlayTimer = null;
    }
}

function updateTimerDisplay() {
    const minutes = Math.floor(gameTimer / 60);
    const seconds = gameTimer % 60;
    const timerElement = document.getElementById('gameTimer');
    if (timerElement) {
        timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
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
            { row: -1, col: 0, name: 'DOWN' },
            { row: 1, col: 0, name: 'UP' },
            { row: 0, col: -1, name: 'RIGHT' },
            { row: 0, col: 1, name: 'LEFT' }
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
    // console.log('=== DFS DEBUG START ===');
    // console.log('Start state:', start);
    // console.log('Goal state:', goal);
    // console.log('Max depth:', maxDepth);

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

    // console.log('Starting DFS search...');

    while (stack.length > 0) {
        const current = stack.pop();
        const currentString = current.state.join(',');
        nodesExplored++;

        // Track max depth reached
        if (current.depth > maxDepthReached) {
            maxDepthReached = current.depth;
            // console.log(`New max depth reached: ${maxDepthReached}`);
        }

        // Log every 1000 nodes
        if (nodesExplored % 1000 === 0) {
            // console.log(`Nodes explored: ${nodesExplored}, Current depth: ${current.depth}, Stack size: ${stack.length}`);
        }

        // Check if we found the goal
        if (currentString === goalString) {
            // console.log('=== SOLUTION FOUND! ===');
            // console.log(`Path length: ${current.path.length}`);
            // console.log(`Nodes explored: ${nodesExplored}`);
            // console.log(`Max depth reached: ${maxDepthReached}`);
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

        // console.log(`Depth ${current.depth}: Found ${neighbors.length} neighbors`);

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
// // Solve puzzle function
// async function solvePuzzle() {
//     if (isPuzzleSolved()) {
//         updateStatus('Puzzle is already solved!');
//         return;
//     }

// const selectedAlgorithm = document.getElementById('algorithmSelect').value;
// updateStatus(`Solving puzzle using ${selectedAlgorithm.toUpperCase()}...`);

// setTimeout(() => {
//     const startTime = performance.now();
//     let result;

//     switch (selectedAlgorithm) {
//     //         case 'astar':
//     //             result = aStarSearch([...currentState], goalState);
//     //             break;
//     //         case 'bfs':
//     //             result = bfsSearch([...currentState], goalState);
//     //             break;
//     //         case 'dfs':
//     //             result = dfsSearch([...currentState], goalState);
//     //             break;
//     //     }

//         const endTime = performance.now();
//         const solveTime = endTime - startTime;

//         if (result.path.length > 0) {
//             displaySolution(result, solveTime);
//             updateStatus(`${result.algorithm} found solution!`);

//             // Automatically start animation
//             setTimeout(() => {
//                 animateSolution(result.path);
//             }, 1000);

//         } else {
//             updateStatus(`${result.algorithm} could not find solution`);
//         }
//     }, 100);
// }
// Solve puzzle function
async function solvePuzzle() {
    if (isPuzzleSolved()) {
        updateStatus('Puzzle is already solved!');
        return;
    }

    // Disable manual play tracking
    manualPlayActive = false;

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
function playAlgorithmAnimation(algorithmName) {
    if (!comparisonResults || !savedShuffledState) {
        updateStatus('Please shuffle and compare algorithms first!');
        return;
    }

    // Find the result for the requested algorithm
    const result = comparisonResults.find(r => r.algorithm === algorithmName);

    if (!result || result.path.length === 0) {
        updateStatus(`${algorithmName} did not find a solution!`);
        return;
    }

    // Reset to original shuffled state
    currentState = [...savedShuffledState];
    updateDisplay();

    // Start animation after a brief delay
    setTimeout(() => {
        animateSolution(result.path);
    }, 500);
}

// Animation function
function animateSolution(solutionPath) {
    if (!solutionPath || solutionPath.length === 0) return;

    let stepIndex = 0;
    const speed = 500; // 500ms between moves

    function nextStep() {
        if (stepIndex === 0) {
            document.getElementById('algorithmLoader').style.display = 'none';
        }

        if (stepIndex >= solutionPath.length) {
            updateStatus('Solution animation completed!');
            return;
        }
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
function displaySolution(result, solveTime) {
    const container = document.getElementById('solutionSteps');

    // Store the solution globally for navigation
    window.currentSolution = result.path;
    window.currentStepIndex = 0;

    let html = `
        <div style="background: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; margin-top: 10px; border-left: 4px solid #4CAF50;">
            <strong>${result.algorithm} Solution Found!</strong><br>
            Steps: ${result.path.length}<br>
            Nodes explored: ${result.nodesExplored}<br>
            Time: ${Math.round(solveTime)}ms
            ${result.timeout ? '<br><span style="color: #FF9800;">⚠️ Search limited to prevent timeout</span>' : ''}
            ${result.depthLimited ? `<br><span style="color: #FF9800;">⚠️ Depth limited to ${result.depthLimited}</span>` : ''}
        </div>

        <div class="current-step" id="currentStepDisplay">
            <strong>Step 1:</strong> Move ${result.path[0]?.move || 'START'}<br>
            <div style="font-size: 12px; margin-top: 10px; font-family: monospace;">
                ${formatStateForDisplay(result.path[0]?.state || result.path[0]?.state)}
            </div>
        </div>

        <div class="solution-navigation">
            <button class="nav-btn" id="prevStepBtn" onclick="navigateStep(-1)">← Previous Step</button>
            <div class="step-indicator">
                
            </div>
            <button class="nav-btn" id="nextStepBtn" onclick="navigateStep(1)">Next Step →</button>
        </div>
    `;

    container.innerHTML = html;
    updateNavigationButtons();
}

function navigateStep(direction) {
    if (!window.currentSolution) return;

    window.currentStepIndex += direction;

    // Clamp to valid range
    if (window.currentStepIndex < 0) window.currentStepIndex = 0;
    if (window.currentStepIndex >= window.currentSolution.length) {
        window.currentStepIndex = window.currentSolution.length - 1;
    }

    updateStepDisplay();
    updateNavigationButtons();
}

function updateStepDisplay() {
    if (!window.currentSolution || !window.currentSolution[window.currentStepIndex]) return;

    const currentStep = window.currentSolution[window.currentStepIndex];
    const stepDisplay = document.getElementById('currentStepDisplay');
    const stepCounter = document.getElementById('stepCounter');

    if (stepDisplay && currentStep) {
        stepDisplay.innerHTML = `
            <strong>Step ${window.currentStepIndex + 1}:</strong> Move ${currentStep.move}<br>
            <div style="font-size: 12px; margin-top: 10px; font-family: monospace;">
                ${formatStateForDisplay(currentStep.state)}
            </div>
        `;
    }

    if (stepCounter) {
        stepCounter.textContent = `Step ${window.currentStepIndex + 1} of ${window.currentSolution.length}`;
    }

    // Safely update the puzzle display
    if (currentStep && currentStep.state && currentStep.state.length === 9) {
        currentState = [...currentStep.state];
        updateDisplay();
    } else {
        console.error('Invalid step state:', currentStep);
    }
}

function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevStepBtn');
    const nextBtn = document.getElementById('nextStepBtn');

    if (prevBtn) prevBtn.disabled = window.currentStepIndex <= 0;
    if (nextBtn) nextBtn.disabled = window.currentStepIndex >= window.currentSolution.length - 1;
}
// Compare all algorithms function
async function compareAllAlgorithms() {
    document.getElementById('algorithmLoader').style.display = 'block';
    document.getElementById('stepCounter').textContent = 'Puzzle Solved';

    if (isPuzzleSolved()) {
        updateStatus('Please shuffle the puzzle first before comparing algorithms!');
        return;
    }

    // Show loading state
    // const compareBtn = document.getElementById('compareBtn');
    // // if (compareBtn) {
    // //     compareBtn.innerHTML = '<span>🔄 Solving...</span>';
    // //     compareBtn.disabled = true;
    // // }

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
        // Store results globally
        comparisonResults = results;

        // Display results
        displayComparisonResults(results);
        updateStatus('Algorithm comparison completed! Choose an algorithm to animate.');

        // Set best solution
        const bestResult = results.find(r => r.algorithm === 'A*' && r.path.length > 0) ||
            results.find(r => r.path.length > 0);
        if (bestResult && bestResult.path.length > 0) {
            solution = bestResult.path;
            currentSolutionStep = 0;
            // document.getElementById('nextStepBtn').disabled = false;
        }

        // // Reset button and show success
        // if (compareBtn) {
        //     compareBtn.innerHTML = '✅ Success!';
        //     setTimeout(() => {
        //         compareBtn.innerHTML = 'Solve Puzzle and Compare';
        //         compareBtn.disabled = false;
        //     }, 2000); // Show success for 2 seconds
        // }

    } catch (error) {
        console.error('Error in comparison:', error);
        updateStatus('Error during algorithm comparison');
    }
}

function displayComparisonResults(results) {
    // Hide the info panel during comparison
    // document.querySelector('.info-panel').style.display = 'none';
    const infoPanel = document.querySelector('.info-panel');
    if (infoPanel) infoPanel.style.display = 'none';

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

    // Show only the best algorithm's solution
    const bestResult = validResults.find(r => r.algorithm === 'A*') || validResults[0];
    if (bestResult && bestResult.path && bestResult.path.length > 0) {
        displaySolution(bestResult, bestResult.time);
    } else {
        document.getElementById('solutionSteps').innerHTML = '';
    }

    // Auto-play the best result, then show manual controls
    if (bestResult && bestResult.path && bestResult.path.length > 0) {
        setTimeout(() => {
            // Reset to shuffled state first
            currentState = [...savedShuffledState];
            updateDisplay();

            // Auto-play best result after brief delay
            setTimeout(() => {
                animateSolution(bestResult.path);
            }, 500);
        }, 1000);
    }
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

// // Play next solution step
// function playNextStep() {
//     if (currentSolutionStep < solution.length) {
//         currentState = [...solution[currentSolutionStep].state];
//         updateDisplay();

//         // Highlight current step
//         document.querySelectorAll('.solution-step').forEach((step, index) => {
//             step.style.background = index === currentSolutionStep ? '#e3f2fd' : 'white';
//         });

//         currentSolutionStep++;

//         if (currentSolutionStep >= solution.length) {
//             document.getElementById('nextStepBtn').disabled = true;
//             updateStatus('Solution completed!');
//         }
//     }
// }

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
    manualPlayStartTime = null;
    stopGameTimer();
    gameTimer = 0;
    updateTimerDisplay();
    updateDisplay();
    updateStatus('Puzzle reset to solved state.');
    // document.getElementById('nextStepBtn').disabled = true;
    // document.getElementById('solutionSteps').innerHTML = '<p>Upload an image and click "Find Solution" to see the optimal path.</p>';

}

function resetButton(){
    currentState = [...savedShuffledState];
    updateDisplay();

    updateStatus('Returned to shuffled state. You can continue solving or click "Find Solution".');
    manualPlayActive = true;
    moveCount = 0;
    const stepCounter = document.getElementById('stepCounter');
    if (stepCounter) stepCounter.textContent = 'Step 0';

}

// Update status
function updateStatus(message) {
    // document.getElementById('gameStatus').textContent = message;
    const gameStatusElement = document.getElementById('gameStatus');
    if (gameStatusElement) gameStatusElement.textContent = message;
}

// // Show success message
function showSuccess() {
    stopGameTimer();

    const existingMessage = document.querySelector('.success-message');
    if (existingMessage) return;

    const message = document.createElement('div');
    message.className = 'success-message';
    message.textContent = `Congratulations! Puzzle solved in ${moveCount} moves and ${Math.floor(gameTimer / 60)}:${(gameTimer % 60).toString().padStart(2, '0')}!`;

    const container = document.querySelector('.left-panel');
    container.insertBefore(message, container.firstChild);

    setTimeout(() => {
        if (message.parentNode) {
            message.parentNode.removeChild(message);
        }
    }, 5000);

    updateStatus('Puzzle solved!');
}

// // Update algorithm info when selection changes
// document.getElementById('algorithmSelect').addEventListener('change', function () {
//     const algorithm = this.value;
//     const titleElement = document.getElementById('algorithmTitle');
//     const descElement = document.getElementById('algorithmDescription');
//     const currentAlgElement = document.getElementById('currentAlgorithm');
//     const heuristicElement = document.getElementById('heuristicDisplay');

//     switch (algorithm) {
//         case 'astar':
//             titleElement.textContent = 'Algorithm: A* with Manhattan Distance Heuristic';
//             descElement.textContent = 'A* uses Manhattan distance heuristic to find optimal solutions efficiently. It combines the actual cost to reach a state with an estimated cost to the goal, guaranteeing the shortest path while exploring fewer nodes than uninformed search.';
//             currentAlgElement.textContent = 'A* Search';
//             heuristicElement.textContent = 'Manhattan Distance';
//             break;
//         case 'bfs':
//             titleElement.textContent = 'Algorithm: Breadth-First Search (BFS)';
//             descElement.textContent = 'BFS explores all nodes at the current depth before moving to nodes at the next depth. It guarantees finding the optimal solution but may explore many more nodes than A*. It uses no heuristic information.';
//             currentAlgElement.textContent = 'BFS';
//             heuristicElement.textContent = 'None (Uninformed)';
//             break;
//         case 'dfs':
//             titleElement.textContent = 'Algorithm: Depth-First Search (DFS) - Limited Depth';
//             descElement.textContent = 'DFS explores as far as possible along each branch before backtracking. For 8-puzzle, we limit the depth to prevent infinite loops. DFS does not guarantee optimal solutions and may find longer paths.';
//             currentAlgElement.textContent = 'DFS (Limited)';
//             heuristicElement.textContent = 'None (Uninformed)';
//             break;
//     }
// });

// Initialize the game
initializePuzzle();
updateStatus('Upload an image to start playing!');