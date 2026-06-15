def build_analysis_prompt(data: dict, focus: str = "general") -> str:
    focus_instructions = {
        "general":     "Provide a balanced set of recommendations covering the biggest opportunities.",
        "retention":   "Focus specifically on reducing churn and re-engaging customers who haven't bought recently.",
        "upsell":      "Focus on increasing average order value and cross-selling to existing customers.",
        "acquisition": "Focus on attracting new customers similar to the top performing segments.",
        "seasonal":    "Focus on capitalising on seasonal trends visible in the data.",
    }
    focus_text = focus_instructions.get(focus, focus_instructions["general"])
    summary    = data.get("summary", {})

    # ── Overview ──────────────────────────────────────────────────────────
    overview_section = f"""
## Dataset Overview
- Total customers: {summary.get("total_customers", "N/A")}
- Total transactions: {summary.get("total_transactions", "N/A")}
- Total revenue: ${summary.get("total_revenue", 0):,.2f}
- Average order value: ${summary.get("avg_order_value", 0):,.2f}
- Date range: {summary.get("date_start", "N/A")} to {summary.get("date_end", "N/A")}
"""

    # ── Segments ──────────────────────────────────────────────────────────
    segments = data.get("segments", [])
    if segments:
        segments_section = "\n## Customer Segments\n"
        for seg in segments:
            segments_section += (
                f"- **{seg['label']}** — {seg['pct_of_customers']}% of customers "
                f"({seg['size']} people), avg spend ${seg['avg_monetary']:,.2f}, "
                f"last purchased {seg['avg_recency_days']} days ago, "
                f"buys {seg['avg_frequency']:.1f}x per period\n"
            )
    else:
        segments_section = "\n## Customer Segments\nNo segmentation data available.\n"

    # ── Association Rules — increased to top 10 ───────────────────────────
    rules = data.get("association_rules", [])
    if rules:
        rules_section = "\n## Purchase Patterns (What customers buy together)\n"
        for rule in rules[:10]:   # was 6, now 10
            ant = " + ".join(rule["antecedents"])
            con = " + ".join(rule["consequents"])
            rules_section += (
                f"- Customers who buy **{ant}** also buy **{con}** "
                f"({rule['confidence']*100:.0f}% of the time, "
                f"{rule['lift']:.1f}x more likely than random)\n"
            )
    else:
        rules_section = "\n## Purchase Patterns\nNo pattern data available.\n"

    # ── Trend Data — NEW: actually pass revenue trends to LLM ─────────────
    trend = data.get("trend_data", {})
    if trend and trend.get("monthly_revenue"):
        monthly = trend["monthly_revenue"]
        mom     = trend.get("mom_growth_pct", 0)
        top_products = trend.get("top_products", [])

        # Find best and worst months
        best  = max(monthly, key=lambda x: x["total_revenue"])
        worst = min(monthly, key=lambda x: x["total_revenue"])

        trend_section = f"""
## Revenue Trends
- Month-over-month growth (latest): {mom:+.1f}%
- Best month: {best['month']} (${best['total_revenue']:,.2f})
- Worst month: {worst['month']} (${worst['total_revenue']:,.2f})
- Revenue range: ${worst['total_revenue']:,.2f} – ${best['total_revenue']:,.2f}
"""
        if top_products:
            trend_section += "\nTop products by revenue:\n"
            for p in top_products[:5]:
                trend_section += (
                    f"- {p['product_name']}: "
                    f"${p['total_revenue']:,.2f} "
                    f"({int(p['total_quantity'])} units)\n"
                )

        if trend.get("category_monthly"):
            cats = list(trend["category_monthly"].keys())
            trend_section += f"\nCategories with monthly data: {', '.join(cats)}\n"
    else:
        trend_section = "\n## Revenue Trends\nNo trend data available.\n"

    # ── Silhouette score explanation — NEW ────────────────────────────────
    silhouette = data.get("silhouette_score")
    if silhouette:
        if silhouette >= 0.5:
            sil_interp = "strong — the customer groups are very distinct"
        elif silhouette >= 0.3:
            sil_interp = "moderate — the segments are reasonably distinct but overlap somewhat"
        else:
            sil_interp = "weak — the segments overlap significantly; treat them as directional"
        sil_note = f"\nNote: Cluster quality (silhouette score {silhouette:.3f}) is {sil_interp}.\n"
    else:
        sil_note = ""

    # ── Full prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a senior marketing strategist with expertise in customer analytics.
Your audience is a marketing manager — not technical. Avoid terms like "silhouette score".
Instead explain what the numbers mean for the business.

{focus_text}

---
{overview_section}
{segments_section}
{rules_section}
{trend_section}
{sil_note}
---

Based on the data above, provide a strategic marketing report with:

1. **Executive Summary** — 2-3 sentences on the biggest insight
2. **Top 3 Recommendations** — specific action, which segment to target, expected impact
3. **One Risk or Caveat** — something to watch out for
4. **Suggested A/B Test** — one concrete experiment to validate top recommendation

Be specific. Reference the actual numbers from the data.
Keep language clear and actionable.
"""
    return prompt