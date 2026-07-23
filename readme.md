# CarterX.ai

> AI-powered customer analytics platform. Upload your sales data. Get customer segments, purchase patterns, revenue trends, geographic insights, and an LLM-generated marketing strategy — automatically.

---

## What is CarterX?

CarterX is a full-stack AI analytics platform built for marketing teams. Companies upload their sales CSV or Excel file and receive a complete customer intelligence report in under a minute — no data science expertise required.

The platform automatically:
- Segments customers into behavioral groups using KMeans clustering on RFM features
- Mines purchase patterns using FP-Growth association rule learning
- Visualizes customer similarity in 2D using t-SNE dimensionality reduction
- Analyzes revenue trends, top products, and geographic distribution
- Generates a written marketing strategy report using an LLM of the user's choice

---

## Architecture

```
User uploads CSV / XLSX
         │
         ▼
┌─────────────────────┐
│   React Frontend    │  localhost:3000
│   (Upload+Dashboard)│
└────────┬────────────┘
         │ HTTP (axios)
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  localhost:8000
│   /api/v1/          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           ML Pipeline (Background)      │
│                                         │
│  preprocessing → segmentation → t-SNE   │
│  → association rules → geo analysis     │
│  → LLM report                           │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   SQLite/PostgreSQL │
│   (Jobs + Insights) │
└─────────────────────┘
```

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| SQLAlchemy (async) | ORM and database sessions |
| SQLite | Local development database |
| Pydantic v2 | Request/response validation |
| Background Tasks | Async ML pipeline execution |

### Machine Learning
| Library | Purpose |
|---|---|
| scikit-learn | KMeans clustering, RobustScaler, silhouette scoring |
| mlxtend | FP-Growth association rule mining |
| pandas + numpy | Data cleaning, RFM feature engineering |
| rapidfuzz | Fuzzy column name matching |
| sklearn.manifold.TSNE | 2D customer visualization |

### LLM Integration
| Provider | Model | Status |
|---|---|---|
| Groq | Llama 3.3 70B Versatile | ✅ Free, default |
| Google Gemini | gemini-2.0-flash | ✅ Free tier |
| OpenAI | gpt-4o-mini | ✅ Paid |
| Anthropic | claude-sonnet | ✅ Paid |

### Frontend
| Technology | Purpose |
|---|---|
| React 18 | Component-based UI |
| React Router | Client-side navigation |
| Recharts | Interactive charts |
| Axios | HTTP client |
| Instrument Serif + Inter | Typography |

---

## Project Structure

```
CarterX.ai/
├── app/
│   ├── main.py                         # FastAPI app entry point, CORS, route registration
│   ├── core/
│   │   └── config.py                   # All settings loaded from .env (Pydantic)
│   ├── db/
│   │   └── session.py                  # Async SQLAlchemy engine, get_db dependency
│   ├── models/
│   │   ├── job.py                      # Job table — status, stage_status, timestamps
│   │   └── insight.py                  # Insight table — all ML results + LLM report
│   ├── schemas/
│   │   ├── job.py                      # JobCreate, JobResponse
│   │   ├── insight.py                  # InsightResponse
│   │   └── report.py                   # ReportRequest, ReportResponse, ModelChoice enum
│   ├── api/
│   │   └── routes/
│   │       ├── health.py               # GET /health, GET /health/detailed
│   │       ├── jobs.py                 # GET /jobs, GET /jobs/{id}
│   │       ├── uploads.py              # POST /uploads — validates, creates job, starts pipeline
│   │       ├── insights.py             # GET /insights/{job_id}
│   │       └── reports.py              # POST /reports/analyze — regenerate with any model/focus
│   └── services/
│       ├── llm.py                      # LLM abstraction — routes to Groq/Gemini/OpenAI/Anthropic
│       ├── prompt_builder.py           # Builds structured prompts from ML outputs
│       ├── pipeline.py                 # Orchestrates all ML stages end-to-end
│       └── ml/
│           ├── preprocessing.py        # Flexible preprocessing — transactional + review datasets
│           ├── segmentation.py         # KMeans + RobustScaler + silhouette optimization
│           ├── association_rule.py     # FP-Growth with product/category mode auto-selection
│           ├── tsne.py                 # t-SNE 2D embedding for cluster visualization
│           └── geo_analysis.py         # Geographic revenue and customer distribution
│
├── frontend/
│   └── src/
│       ├── App.js                      # React Router setup
│       ├── services/
│       │   └── api.js                  # Axios client, all API calls
│       ├── hooks/
│       │   └── useJobPolling.js        # Polls job status every 3s, stops on completion
│       └── pages/
│           ├── UploadPage.js           # Upload form + pipeline progress display
│           ├── DashboardPage.js        # Sidebar layout, tab routing
│           └── dashboard/
│               ├── VisualizationsTab.js  # Revenue charts, segment donut, top products
│               ├── ClustersTab.js        # Segment cards + RFM comparison bars
│               ├── SimulationTab.js      # t-SNE canvas map, RFM histograms, what-if sliders
│               ├── RulesTab.js           # Sortable/searchable association rules table
│               ├── ReportTab.js          # LLM report with model + focus selector
│               ├── StatsTab.js           # Full dataset statistics breakdown
│               └── GeographyTab.js       # Regional revenue, concentration, drill-down
│
├── .env                                # API keys and config (never commit)
├── .env.example                        # Template for required environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ML Pipeline — How it works

Every uploaded file goes through 6 automatic stages. The frontend polls for progress every 3 seconds.

### Stage 1 — Preprocessing
- Detects dataset type: `transactional` (standard retail), `review` (Amazon-style), or `catalog`
- Fuzzy column name matching — maps `CustomerID`, `cust_id`, `user_id` → all to `customer_id`
- Cleans currency symbols (₹, $, €), commas in numbers, percentage signs
- Handles Amazon-style packed rows (multiple user IDs per row)
- Engineers RFM features: **Recency**, **Frequency**, **Monetary**
- Scales with `RobustScaler` + log-transform for skewed spend data

### Stage 2 — Segmentation
- Automatically finds optimal K (2–8 clusters) by maximizing silhouette score
- Assigns human-readable labels: Champions, Loyal Customers, At Risk, Hibernating, New Customers, etc.
- Returns labelled RFM dataframe for downstream t-SNE

### Stage 3 — t-SNE
- Reduces 3D RFM space to 2D for visual cluster mapping
- Perplexity auto-adjusted for small datasets
- Each point carries customer metadata for hover tooltips on the canvas

### Stage 4 — Association Rules
- FP-Growth algorithm on transaction baskets
- Auto-selects product-level or category-level mining based on basket density
- Automatically relaxes support/confidence thresholds for sparse data
- Returns top 20 rules sorted by lift

### Stage 5 — Geographic Analysis
- Detects any geographic column (country, region, state, city, province…)
- Computes revenue, customers, avg order value per region
- Calculates HHI (Herfindahl-Hirschman Index) for market concentration
- Month-over-month growth per region, top products per region

### Stage 6 — LLM Report
- Builds a structured prompt from all ML outputs — segments, rules, trends, geo insights, silhouette score interpretation
- Sends to selected LLM (default: Groq Llama 3.3 70B — free)
- Returns a formatted markdown strategy report with executive summary, recommendations, risk, and A/B test suggestion

---

## Dataset Support

CarterX accepts any CSV or XLSX file. It automatically maps column names using fuzzy matching — you don't need to rename anything.

### Transactional datasets (standard)
Works with any file containing: customer ID, product, quantity, price, date

Examples: UCI Online Retail, internal POS exports, Shopify exports

### Review / Rating datasets
Works with Amazon-style files containing: user ID, product ID, rating, price

Examples: Amazon product reviews dataset (Kaggle)

### What columns are recognized

| Your column name | Maps to |
|---|---|
| `CustomerID`, `cust_id`, `user_id`, `buyer_id` | `customer_id` |
| `InvoiceNo`, `order_id`, `review_id` | `transaction_id` |
| `Description`, `product_title`, `item_name` | `product_name` |
| `StockCode`, `asin`, `sku` | `product_id` |
| `UnitPrice`, `discounted_price`, `selling_price` | `price` |
| `InvoiceDate`, `order_date`, `sale_date` | `date` |
| `Country`, `region`, `state`, `city`, `territory` | `region` (geo) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Conda (recommended)

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/CarterX.ai.git
cd CarterX.ai

conda create --name Carter python=3.11
conda activate Carter
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
ENVIRONMENT=development
APP_NAME=CarterX.ai

# LLM API keys — only need at least one
GROQ_API_KEY=your_groq_key_here         # Free at console.groq.com
GEMINI_API_KEY=your_gemini_key_here     # Free at aistudio.google.com
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

DEFAULT_LLM=groq

# Upload limits
MAX_UPLOAD_SIZE_MB=50
MIN_ROWS_REQUIRED=100
```

Get your free Groq API key at: https://console.groq.com/

### 3. Start the backend

```bash
python -m uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 4. Start the frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## API Reference

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/uploads/` | Upload CSV/XLSX, start pipeline |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and stage progress |
| `GET` | `/api/v1/jobs/` | List all jobs |
| `GET` | `/api/v1/insights/{job_id}` | Get all ML results and LLM report |
| `POST` | `/api/v1/reports/analyze` | Regenerate report with different model/focus |

### Example: Upload a file

```bash
curl -X POST http://localhost:8000/api/v1/uploads/ \
  -F "file=@sales_data.csv"
```

Response:
```json
{
  "id": "abc-123",
  "status": "pending",
  "filename": "sales_data.csv",
  "row_count": 5000,
  "stage_status": {
    "preprocessing": "pending",
    "segmentation": "pending",
    "association_rules": "pending",
    "geo_analysis": "pending",
    "llm_report": "pending"
  }
}
```

### Example: Regenerate report with different model and focus

```bash
curl -X POST http://localhost:8000/api/v1/reports/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "abc-123",
    "model": "gemini",
    "focus": "retention"
  }'
```

Available models: `groq`, `gemini`, `openai`, `anthropic`
Available focus modes: `general`, `retention`, `upsell`, `acquisition`, `seasonal`

---

## Dashboard Tabs

| Tab | What you see |
|---|---|
| **Visualizations** | Monthly revenue bar chart, customer segment donut, top products, category trends |
| **Segments** | Customer group cards with RFM profile, comparison bars |
| **Simulation** | t-SNE 2D cluster map (canvas-drawn), RFM histograms, what-if campaign simulator with real-time revenue projection |
| **Rules** | Full association rules table — searchable, sortable by lift/confidence/support |
| **AI Report** | LLM-generated strategy report with model selector and focus selector |
| **Geography** | Regional revenue breakdown, HHI concentration score, growth by region, product drill-down per region |
| **Statistics** | Complete dataset statistics — customers, transactions, revenue, segments, rules, category breakdown |

---

## What-If Simulation

The simulation panel lets marketing teams model campaign impact without re-running the pipeline:

- **Target segment** — select one or all customer segments
- **Discount %** — adjust 0–50% discount applied
- **Time horizon** — 30 to 365 days

The model uses price elasticity (−1.5) to estimate volume uplift from discounts, then projects revenue impact vs baseline. Results update in real time as sliders move.

---

## Key Technical Decisions

**Why RobustScaler + log-transform for RFM?**
Retail spend data is almost always right-skewed — a few whale customers spend 100x the median. StandardScaler gets dragged by outliers, compressing the 95% of normal customers into a tiny range. RobustScaler uses median/IQR instead of mean/std, making clusters reflect genuine behavioral differences rather than outlier distortion.

**Why FP-Growth over Apriori?**
FP-Growth builds a compressed prefix tree and mines patterns without repeatedly scanning the dataset. Significantly faster on large transaction files (100k+ rows).

**Why t-SNE over PCA for cluster visualization?**
t-SNE preserves local neighborhood structure — customers who are similar in RFM space stay close together in 2D. PCA preserves global variance which can obscure cluster boundaries. The tradeoff is t-SNE distances between clusters are not meaningful, only within-cluster proximity is.

**Why background tasks instead of Redis/Celery?**
FastAPI's built-in `BackgroundTasks` is sufficient for a single-server deployment and requires zero infrastructure. For production at scale, the pipeline worker can be migrated to a Celery/Redis queue with no changes to business logic.

**Why store Plotly specs as JSON instead of images?**
All chart data is stored as raw arrays in the database. The frontend renders them fresh on every load, allowing filtering, resizing, and theme changes without hitting the backend again.

---

## Environment Variables Reference

```env
# App
ENVIRONMENT=development           # development | production
APP_NAME=CarterX.ai

# LLM providers (add whichever you have)
GROQ_API_KEY=                     # Free — console.groq.com
GEMINI_API_KEY=                   # Free — aistudio.google.com/app/apikey
OPENAI_API_KEY=                   # Paid — platform.openai.com
ANTHROPIC_API_KEY=                # Paid — console.anthropic.com
DEFAULT_LLM=groq                  # Which model to use by default

# Upload constraints
MAX_UPLOAD_SIZE_MB=50
MIN_ROWS_REQUIRED=100
```

---

## Known Limitations

- SQLite is used for local development. For production, switch `DATABASE_URL` to PostgreSQL — no code changes required.
- t-SNE can be slow on very large datasets (100k+ customers). Consider sampling to 10k customers for the visualization on large files.
- The simulation uses a simplified price elasticity model (−1.5). Real elasticity varies by product category, season, and customer segment.
- Geographic analysis requires a column explicitly named country, region, state, city, or similar. Datasets without a location column will show an empty state on the Geography tab.

---

## Future Scope

- **GCP deployment** — Cloud Run for backend and frontend, Cloud SQL for PostgreSQL, Secret Manager for API keys, Cloud Build for CI/CD
- **Prophet-based forecasting** — per-category time series forecasting with 90-day horizon and confidence bands
- **Cohort analysis** — customer retention curves by acquisition month
- **Email campaign integration** — export segment membership to Mailchimp/SendGrid via API
- **Real-time WebSocket updates** — replace polling with push notifications for pipeline progress
- **Multi-tenant support** — user accounts, separate data isolation per organization

---

## Built With

This project was built as a portfolio project demonstrating:
- End-to-end ML system design with a production-quality pipeline
- FastAPI backend with async patterns, dependency injection, and background tasks
- Multi-provider LLM integration with a clean abstraction layer
- React frontend with real-time polling, interactive charts, and a canvas-drawn visualization
- Flexible data ingestion that handles multiple real-world dataset formats

---

## License

MIT License — free to use, modify, and distribute.