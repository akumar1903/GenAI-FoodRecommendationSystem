# GenAI-FoodRecommendationSystem
Food recommendation system using ChromaDB and HuggingFace model. 

# Food Recommendation System using ChromaDB & HuggingFace

This Node.js project uses [ChromaDB](https://docs.trychroma.com/) for vector storage and [Hugging Face](https://huggingface.co/) models for text embeddings and zero-shot classification to build a **smart food recommendation system**.

---

## Features

- Uses **sentence-transformers/all-MiniLM-L6-v2** to generate text embeddings.
- Uses **zero-shot classification** to extract dietary and cuisine preferences from queries.
- Stores food item data in a Chroma collection.
- Returns **top 5 similar food recommendations** based on user input.
- Supports **filters** like "vegan", "vegetarian", "indian", "japanese", etc.

---

## Project Structure

```
.
├── FoodDataSet.js         # Sample food data (must be in CommonJS export)
├── index.js               # Main code
├── README.md              # This file
```

---

## Installation

1. **Clone this repository**:

```bash
git clone https://github.com/your-username/food-recommendation-app.git
cd food-recommendation-app
```

2. **Install dependencies**:

```bash
npm install chromadb @huggingface/inference
```

> ⚠️ If you don’t have ChromaDB running, follow [official setup instructions](https://docs.trychroma.com/getting-started) to start it locally or via Docker.

---

## Download Dataset (Optional)

If you don’t already have `FoodDataSet.js`, download it:

```bash
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/CBCVVX4wJjXG64DKYMVi1w/FoodDataSet.js
```

---

## Hugging Face Token

Generate a free token from [Hugging Face](https://huggingface.co/settings/tokens) and replace this line in `index.js`:

```js
const hf = new HfInference("your_hf_token_here");
```

---

## Running the App

```bash
node index.js
```

---

## Example Query

```js
const query = "I want to eat food with chicken and masala";
```

This query will:
- Use Hugging Face to extract diet/cuisine type (e.g., non-vegetarian, Indian).
- Use semantic search to return the 5 most similar food items based on description and ingredients.

---

## Output Example

```
Extracted Filter Criteria (Diet): { diet: 'non-vegetarian', cuisine: null }
Top 1 Recommended Food Name: Chicken Tikka Masala
Top 2 Recommended Food Name: Butter Chicken Curry
...
```

---

## Notes

- The app requires **Node.js v16+**.
- The code uses **CommonJS** modules.
- Make sure ChromaDB server is running and accessible.

---

## License

MIT License. Feel free to fork and build on top of it!
