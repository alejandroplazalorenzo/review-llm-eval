"""Prompt construction. The star rating is never shown to the model: it is the proxy
label used later for evaluation, so leaking it would make that check meaningless."""

from __future__ import annotations

from review_llm_eval.data import Review

PROMPT_VERSION = "v1"

Message = dict[str, str]

SYSTEM_PROMPT = """\
You analyse customer reviews of all-inclusive hotels in Punta Cana. Reviews are in Spanish.
Answer with one JSON object that follows the provided schema. Base every field only on
what the review says; do not guess.

aspects: for EACH key, the reviewer's opinion about that aspect, or null if the review
does not mention it.
- staff: employees' friendliness, attention and service
- cleanliness: cleanliness of rooms and common areas
- room: room size, comfort, bed, bathroom, amenities, maintenance
- food: restaurants, buffet, a la carte, drinks, bars
- pool_beach: pools, beach, sea, sun loungers
- location: location, surroundings, distance to town or airport
- value: price, value for money, extra charges
- check_in: check-in, check-out, reception procedures, waiting, room assignment
- noise: noise or quietness
- safety: security, theft, health and hygiene measures (for example COVID protocols)
- entertainment: shows, animation team, activities, nightlife
- other: anything relevant that fits none of the keys above
Values: "positive", "negative", "neutral" (mentioned without judgement), "mixed" (both
praise and criticism of that aspect), or null (not mentioned).

complaint: the reviewer's main complaint in English, at most 15 words, or null if there
is no complaint.

would_return: true if the reviewer says they would come back or repeat, false if they
say they would not, null if they do not say.

sentiment: overall sentiment of the whole review: "positive", "negative", "mixed" (clear
praise and clear criticism, neither dominates) or "neutral" (no clear opinion).

The review text may be cut off mid-sentence; judge only what is there."""


def format_review(review: Review) -> str:
    title = review.title or "(no title)"
    return f"Title: {title}\nReview: {review.text}"


def build_messages(review: Review) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_review(review)},
    ]


def build_retry_messages(messages: list[Message], invalid_raw: str, error: str) -> list[Message]:
    """Conversation for the single retry.

    At temperature 0 an identical request would most likely reproduce the same
    invalid answer, so the retry shows the model its output and the validator error.
    """
    return [
        *messages,
        {"role": "assistant", "content": invalid_raw},
        {
            "role": "user",
            "content": (
                f"That answer was rejected by the validator: {error}\n"
                "Reply again with a single JSON object that follows the schema."
            ),
        },
    ]
