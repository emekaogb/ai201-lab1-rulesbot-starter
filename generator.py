from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    # Filter chunks by distance threshold — only use relevant matches
    distance_threshold = 0.5
    relevant_chunks = [c for c in retrieved_chunks if c["distance"] < distance_threshold]

    if not relevant_chunks:
        return (
            "I don't have any information on this in the loaded rule books. "
            "Try rephrasing your question or ask about a different game."
        )

    # Format context with game labels and distance scores
    context = ""
    for chunk in relevant_chunks:
        context += f"[{chunk['game']}, relevance: {chunk['distance']:.2f}]\n{chunk['text']}\n\n"

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant answering questions about board game rules.

CRITICAL: You must ONLY answer based on the rule text provided below. Do not use your general knowledge about games.
- If the answer is found in the rules, cite which game it comes from.
- If the answer is NOT in the provided rules, respond with: "I don't have that information in the loaded rule books."
- Never make up or infer rules that aren't explicitly stated in the provided text.

The rules are labeled by game and include relevance scores. Lower scores mean more relevant matches."""
        },
        {
            "role": "user",
            "content": f"""Question: {query}

Retrieved rules from the rulebooks:

{context}

Answer using ONLY the rules provided above. Always cite which game the rule comes from. If the information is not in the rules, say so clearly."""
        }
    ]

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content
