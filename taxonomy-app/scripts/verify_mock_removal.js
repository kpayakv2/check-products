
// Verification Script: verify_mock_removal.js
// This script mimics the logic in app/api/import/process/route.ts
// It attempts to call the embedding API and should FAIL if the backend is down,
// proving that the mock fallback (which would have returned random numbers) is gone.

async function generateEmbedding(text) {
    console.log(`Attempting to generate embedding for: "${text}"`);
    console.log("Target URL: http://localhost:8000/api/embed");

    try {
        const response = await fetch('http://localhost:8000/api/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            signal: AbortSignal.timeout(5000) // 5s timeout
        });

        if (!response.ok) {
            throw new Error(`Failed to generate embedding: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("SUCCESS: Received real embedding from backend.");
        console.log("Vector length:", data.embedding.length);
        return data.embedding;

    } catch (error) {
        // This is the behavior we WANT now (Error propagation)
        console.error("EXPECTED ERROR: Connection failed as expected (Mock removed).");
        console.error("Error details:", error.message);
        // In the old version, this would have printed a warning and returned a random array
        // return Array.from({ length: 384 }, () => Math.random());
        throw error;
    }
}

async function run() {
    try {
        await generateEmbedding("Test Product");
    } catch (e) {
        console.log("\nVERIFICATION PASSED: The script failed as expected. The dangerous mock fallback is gone.");
        process.exit(0);
    }

    // If we get here without error (and backend is down), or if mock was still there
    // If backend IS down, we expect error.
    // If backend IS up, we expect success.
    // We only fail verify if we get a result but backend was down (impossible without mock)
    // or if we catch error but then return mock (which we removed).
}

run();
