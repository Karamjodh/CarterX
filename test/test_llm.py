import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.services.llm import generate_report
from app.services.prompt_builder import build_analysis_prompt

# Fake ML data — same structure the real pipeline will produce
SAMPLE_DATA = {
    "summary": {
        "total_customers":    1243,
        "total_transactions": 4821,
        "total_revenue":      487650.00,
        "avg_order_value":    101.15,
        "date_start":         "2024-01-01",
        "date_end":           "2024-12-31",
    },
    "segments": [
        {
            "label":              "Champions",
            "size":               312,
            "pct_of_customers":   25.1,
            "avg_monetary":       820.50,
            "avg_recency_days":   8,
            "avg_frequency":      6.2,
        },
        {
            "label":              "Loyal Customers",
            "size":               445,
            "pct_of_customers":   35.8,
            "avg_monetary":       340.20,
            "avg_recency_days":   22,
            "avg_frequency":      3.1,
        },
        {
            "label":              "At Risk",
            "size":               280,
            "pct_of_customers":   22.5,
            "avg_monetary":       210.80,
            "avg_recency_days":   67,
            "avg_frequency":      1.4,
        },
        {
            "label":              "Lost Customers",
            "size":               206,
            "pct_of_customers":   16.6,
            "avg_monetary":       95.30,
            "avg_recency_days":   180,
            "avg_frequency":      0.8,
        },
    ],
    "association_rules": [
        {
            "antecedents": ["Laptop"],
            "consequents": ["Mouse", "Keyboard"],
            "confidence":  0.81,
            "lift":        3.2,
        },
        {
            "antecedents": ["Phone"],
            "consequents": ["Phone Case"],
            "confidence":  0.74,
            "lift":        2.8,
        },
        {
            "antecedents": ["Running Shoes"],
            "consequents": ["Sports Socks", "Water Bottle"],
            "confidence":  0.68,
            "lift":        2.1,
        },
    ],
    "forecasts": {
        "Electronics": {"trend_pct":  18.5},
        "Fashion":     {"trend_pct": -11.2},
        "Sports":      {"trend_pct":   6.3},
    },
}


async def test():

    # ── Test 1: Check what the prompt looks like ───────────────────────────
    print("=" * 60)
    print("PROMPT PREVIEW")
    print("=" * 60)
    prompt = build_analysis_prompt(SAMPLE_DATA, focus="retention")
    print(prompt)

    # ── Test 2: Send it to Groq and see the response ───────────────────────
    print("\n" + "=" * 60)
    print("LLM RESPONSE")
    print("=" * 60)
    result = await generate_report(prompt, model="groq")
    print(f"Model used:    {result['model_used']}")
    print(f"Input tokens:  {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"\n{result['text']}")


asyncio.run(test())