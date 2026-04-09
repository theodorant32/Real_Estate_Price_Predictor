# Propra

I built this because I'm 23 and honestly terrified I'll never be able to afford a home in Vancouver.

Every time I check Zillow or Realtor.ca, another property goes up by $100K. My savings can't keep up with that. So instead of just doom-scrolling listings, I decided to do what any stubborn CS person would do: **build my own AI to make sense of this mess**.

---

## What This Actually Does

This isn't another "predict housing prices" tutorial project. This is a full platform that answers the questions I actually care about:

| Question | Tab |
|----------|-----|
| "Will this condo be worth more in a year?" | 🔮 Price Predictor |
| "Where should I actually look?" | 🗺️ Market Heatmap |
| "Should I buy or just keep renting?" | 💰 ROI Calculator |
| "What if rates go up another 2%?" | 📈 Scenario Simulator |
| "What property fits my situation?" | 🎯 My Recommendations |
| "Are there any deals left?" | 💎 Hidden Gems |
| "What actually worked for others?" | 📚 Case Studies |

There's also an AI chatbot if you prefer asking questions naturally, and a neighborhood analysis tool that scores walkability and transit access.

---

## Why I Made This

The Canadian housing market feels broken right now. Here's what I'm dealing with:

- **Vancouver detached homes**: ~$2M (lol)
- **My savings**: Maybe 15% down on a condo if I'm lucky
- **Mortgage rates**: Went from 2% to 5%+ in two years
- **Everyone's advice**: "Just wait for the crash" or "Buy now before it's worse"

Nobody actually knows what's happening. So I built something that at least gives me **data-backed uncertainty** instead of **confident guessing**.

---

## The Tech (For Recruiters Who Might Scroll Past)

**ML Stack:**
- Stacked ensemble (XGBoost + LightGBM + Ridge)
- SHAP explainability (so I know WHY it says Calgary is better than Vancouver right now)
- Monte Carlo simulations for uncertainty ranges
- NetworkX graph modeling for neighborhood amenity scoring

**Data Pipeline:**
- Weekly automated retraining via GitHub Actions
- Falls back to historical appreciation rates when data sources fail
- Market regime detection (hot/warm/cooling/cold)

**Deployment:**
- Streamlit frontend
- Railway hosting (free tier, so it sleeps - first load takes ~30s)

---

## Try It

**Live:** [https://real-estate-production-cd36.up.railway.app](https://real-estate-production-cd36.up.railway.app)

Or run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## What I Learned

1. **Calgary currently beats Vancouver** for ROI (5% predicted appreciation vs 3%). This surprised me.
2. **Condos in Vancouver have terrible rental yields** (~3.5% cap rate). You're banking entirely on appreciation.
3. **The 1% rule** (monthly rent = 1% of price) is basically dead in Vancouver. A $750K condo renting for $2,600 is 0.35%.
4. **Graph-based neighborhood scoring** is overkill but I wanted to learn NetworkX.

---

## Not Financial Advice

This is a learning project, not advice from a licensed advisor. I'm not responsible if you lose money. The model can be wrong. The data can be stale. The market can stay irrational longer than either of us can stay solvent.

If you're actually making a 7-figure decision, maybe talk to someone who gets paid to know this stuff.

---

## What's Next

- [ ] Scrape actual listings instead of just benchmark data
- [ ] Add image analysis for listing photos (is that kitchen actually modern or just well-lit?)
- [ ] Connect to MLS API if I can get access
- [ ] Maybe add Toronto and Montreal data

---

## License

MIT. Use it, break it, learn from it.

---

*Built during a quarter-life crisis by someone who checks Realtor.ca more often than Instagram.*
