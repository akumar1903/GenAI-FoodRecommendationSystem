const { ChromaClient } = require("chromadb");
const { HfInference } = require("@huggingface/inference");
const foodItems = require('./FoodDataSet.js');

const client = new ChromaClient();
const hf = new HfInference("Your Hugging Face Key");
const collectionName = "food_collectionzahira";

// Generate embeddings for a list of texts
async function generateEmbeddings(texts) {
    const results = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: texts,
    });
    return results;
}

// Classify input text with candidate labels using Zero-Shot Classification
async function classifyText(text, labels) {
    const response = await hf.zeroShotClassification({
        model: "facebook/bart-large-mnli",
        inputs: text,
        parameters: { candidate_labels: labels }
    });
    return response;
}

// Extract dietary or cuisine filter criteria based on the query
async function extractFilterCriteria(query) {
    const criteria = { diet: null, cuisine: null };

    const dietLabels = ["vegan", "non-vegan", "vegetarian", "non-vegetarian", "pescatarian", "omnivore", "paleo", "ketogenic"];
    const cuisineLabels = ["chinese", "indian", "japanese"];

    const dietResult = await classifyText(query, dietLabels);

    if (dietResult?.labels?.length > 0 && dietResult?.scores?.length > 0) {
        const highestDietScoreLabel = dietResult.labels[0];
        const dietScore = dietResult.scores[0];

        if (dietScore > 0.8) {
            criteria.diet = highestDietScoreLabel;
            console.log('Extracted Filter Criteria (Diet):', criteria);
            return criteria;
        }
    }

    const cuisineResult = await classifyText(query, cuisineLabels);
    if (cuisineResult?.labels?.length > 0 && cuisineResult?.scores?.length > 0) {
        const highestCuisineScoreLabel = cuisineResult.labels[0];
        const cuisineScore = cuisineResult.scores[0];

        if (cuisineScore > 0.8) {
            criteria.cuisine = highestCuisineScoreLabel;
        }
    }

    console.log('Extracted Filter Criteria:', criteria);
    return criteria;
}

// Perform a similarity search with optional filter criteria
async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);

        const results = await collection.query({
            collection: collectionName,
            queryEmbeddings: queryEmbedding,
            n: 5,
        });

        if (!results || !results.ids || results.ids.length === 0) {
            console.log(`No food items found similar to "${queryTerm}"`);
            return [];
        }

        let topFoodItems = results.ids[0].map((id, index) => {
            const food = foodItems.find(item => item.food_id.toString() === id);
            return food ? {
                id,
                score: results.distances[0][index],
                food_name: food.food_name,
                food_description: food.food_description
            } : null;
        }).filter(Boolean);

        return topFoodItems.sort((a, b) => a.score - b.score);
    } catch (error) {
        console.error("Error during similarity search:", error);
        return [];
    }
}

// Main runner
async function main() {
    const query = "I want to each food with chicken and masala";

    try {
        const collection = await client.getOrCreateCollection({ name: collectionName });

        // Ensure unique food IDs
        const uniqueIds = new Set();
        foodItems.forEach((food, index) => {
            while (uniqueIds.has(food.food_id.toString())) {
                food.food_id = `${food.food_id}_${index}`;
            }
            uniqueIds.add(food.food_id.toString());
        });

        // Create input texts and generate embeddings
        const foodTexts = foodItems.map((food) =>
            `${food.food_name}. ${food.food_description}. Ingredients: ${food.food_ingredients.join(", ")}`
        );
        const embeddingsData = await generateEmbeddings(foodTexts);

        await collection.add({
            ids: foodItems.map((food) => food.food_id.toString()),
            documents: foodTexts,
            embeddings: embeddingsData,
        });

        const filterCriteria = await extractFilterCriteria(query);

        const initialResults = await performSimilaritySearch(collection, query, filterCriteria);
        initialResults.slice(0, 5).forEach((item, index) => {
            console.log(`Top ${index + 1} Recommended Food Name: ${item.food_name}`);
        });

    } catch (error) {
        console.error("Error:", error);
    }
}

main();
