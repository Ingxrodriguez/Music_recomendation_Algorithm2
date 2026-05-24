# 🎵 Music Recommendation Algorithm

> *What if an algorithm could feel music the way you do?*

---

## The Story

Imagine you are listening to Cigarettes After Sex at 2am. The slow guitar, the melancholic voice, the feeling of something you can't quite name. You want more of that — not just the same genre, not just the same artist — but that **same feeling**.

That is exactly what this project is about.

As a data scientist at a music startup that just secured its Series B, I was tasked with building an unsupervised recommendation algorithm that groups songs by their **lyrical mood and emotional content** — not by genre, not by artist, but by how they *feel*.

---

## What I Found

The data tells an interesting story. Hip hop stands out immediately — it scores way higher on obscene content than any other genre, and this pattern holds across all time periods. Modern songs in general have gotten more explicit over the decades, especially pop which has changed a lot since the 1950s.

I also noticed that features like `romantic`, `violence`, and `sadness` rarely overlap. Songs tend to belong to one emotional theme, not many. A sad song is rarely explicit. A romantic song is rarely violent. Music is surprisingly consistent in its feelings.

So my hypothesis was simple — **group songs by how they feel, not what they are.**

---

## The Pipeline

```
Raw Data (28,362 songs, 1950–2019)
        ↓
Exploratory Data Analysis
        ↓
Drop metadata + redundant features
        ↓
Log transform len (fix skew)
        ↓
StandardScaler (same scale for everyone)
        ↓
PCA (15 features → 12 components, 90.5% variance retained)
        ↓
KMeans Clustering (k=10, chosen by Elbow + Silhouette Score)
        ↓
Recommendations
```

---

## The Clusters

After training KMeans with k=10, the algorithm found these natural groups:

| Cluster | Mood | Dominant Feature |
|---|---|---|
| 0 | 🎤 Explicit | obscene = 0.46 |
| 1 | 🥊 Dark & Aggressive | violence + shake the audience |
| 2 | ⚔️ Violent | violence = 0.43 |
| 3 | 🎸 Music About Music | music = 0.41 |
| 4 | ⛪ Spiritual | family/gospel = 0.20 |
| 5 | 😢 Melancholic | sadness = 0.43 |
| 6 | 💕 Dating | dating = 0.23 |
| 7 | 🌍 Philosophical | world/life = 0.43 |
| 8 | 🌙 Nocturnal | night/time = 0.40 |
| 9 | ❤️ Romantic | romantic = 0.40 |

Genres do not define the clusters directly — but they show up naturally inside them. Hip hop in the explicit cluster. Blues in the sad one. Gospel in the spiritual one.

---

## The Results

I tested the algorithm on 10 user songs and here is what happened:

- 🎵 **"Eso Beso"** by Paul Anka → Romantic cluster ✅
- 🎵 **"Your Cheating Heart"** by Jerry Lee Lewis → Dating cluster ✅
- 🎵 **"Railway and Gun"** by Taste → Melancholic cluster ✅

But the most interesting result? Cluster 8 grouped **Dennis Brown (reggae)**, **Rage Against the Machine (rock)**, and **Randy Travis (country)** together. Three completely different genres — same lyrical energy. That is the beauty of unsupervised learning. It finds patterns that are not obvious to the human eye.

---

## Tech Stack

- **Python 3.12**
- **pandas, numpy** — data manipulation
- **scikit-learn** — StandardScaler, PCA, KMeans, Silhouette Score
- **matplotlib, seaborn** — visualizations
- **pickle** — model persistence

---

## Project Structure

```
📁 Music_Recommendation_Algorithm
├── 📓 EDA.ipynb
├── 📓 Data_Cleaning.ipynb
├── 📓 Model_K.ipynb
├── 📓 User_Prediction.ipynb
├── 📄 Questions_Report.md
├── 📄 README.md
└── 📁 Dataset/
    ├── Large_data.csv
    └── Small_data.csv
```

---

## Key Takeaways

The algorithm does not care about genre. It groups songs by **how they feel lyrically** — and the recommendations make intuitive sense because of it. No need to tell the algorithm the same thing twice. No need to hardcode genres. Just let the data speak.

> *"If you liked Cigarettes After Sex, here is Lana Del Rey. You're welcome."*

---

## Dataset

Music Dataset: Lyrics and Metadata from 1950 to 2019 — a real-world dataset of 28,362 songs with lyrical scores across 15 emotional and thematic dimensions.
