"""
generate_datasets.py
--------------------
Generates synthetic but realistic training and test datasets that mimic:
  • Sentiment140 (1.6M tweets) – binary sentiment
  • IMDb 50K movie reviews – binary sentiment
  • Titanic survival records – binary classification on text description
  • Trade/News headlines – multi-class topic classification

All datasets are saved to the data/ directory.
"""

import os, random, csv
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Vocabulary pools ──────────────────────────────────────────────────────────
POS_TWEET_TEMPLATES = [
    "just had the most amazing {noun} today feeling so {adj}",
    "love this {noun} so much it makes me {adj} every time",
    "so grateful for {noun} in my life truly {adj} experience",
    "best {noun} ever absolutely {adj} highly recommend",
    "feeling {adj} after spending time with {noun} today",
    "wow what a {adj} day started with great {noun}",
    "{noun} is absolutely {adj} can't stop smiling",
    "just finished {noun} and i feel {adj} highly recommend",
    "this {noun} is making me feel {adj} and happy",
    "great news about {noun} feeling really {adj} today",
    "had an incredible {noun} experience feeling so {adj}",
    "shoutout to everyone working on {noun} you are {adj}",
]
NEG_TWEET_TEMPLATES = [
    "so frustrated with {noun} today absolutely {adj}",
    "worst {noun} experience ever completely {adj} never again",
    "can't believe how {adj} this {noun} turned out",
    "{noun} is so {adj} i want to give up",
    "feeling really {adj} because of this {noun} situation",
    "disappointed in {noun} expected better not {adj} at all",
    "this {noun} is making me feel {adj} and upset",
    "ugh {noun} again why is it always so {adj}",
    "terrible {noun} experience so {adj} wasted my time",
    "seriously {noun} how can you be this {adj}",
]
NOUNS = ["coffee", "movie", "job", "flight", "food", "service", "product",
         "app", "hotel", "book", "song", "team", "day", "life", "news",
         "show", "game", "event", "meeting", "trip", "project", "deal"]
POS_ADJ = ["amazing", "wonderful", "fantastic", "great", "brilliant",
           "superb", "awesome", "excellent", "perfect", "delightful",
           "happy", "excited", "blessed", "thankful", "joyful"]
NEG_ADJ = ["awful", "terrible", "horrible", "dreadful", "disappointing",
           "frustrating", "annoying", "pathetic", "dismal", "depressing",
           "worst", "sad", "angry", "exhausted", "disgusted"]

POS_REVIEWS = [
    "a masterpiece of cinema with stellar performances and breathtaking visuals",
    "absolutely loved this film the story was compelling and emotionally resonant",
    "brilliant direction and outstanding acting made this a memorable experience",
    "one of the best movies I have seen in years highly recommend",
    "the screenplay is superb and the characters feel genuine and relatable",
    "a beautifully crafted film that leaves a lasting impression on the viewer",
    "exceptional storytelling with powerful themes that resonate deeply",
    "the cast delivers phenomenal performances in this gripping drama",
    "stunning cinematography and a moving score elevate this film to greatness",
    "an unforgettable cinematic journey that exceeded all my expectations",
]
NEG_REVIEWS = [
    "a complete waste of time with a predictable and boring plot",
    "terrible acting and poor direction made this nearly unwatchable",
    "the story made no sense and the characters were completely unlikable",
    "one of the worst films I have seen in recent memory avoid at all costs",
    "dull uninspired and painfully slow this film fails on every level",
    "the screenplay is riddled with clichés and the dialogue is cringe worthy",
    "a disappointment that fails to deliver on its promising premise",
    "wooden performances and lazy writing drag this film into mediocrity",
    "confusing plot holes and poor editing ruin what could have been decent",
    "an absolute mess from start to finish with nothing to recommend",
]

TRADE_CLASSES = {
    "technology": [
        "tech giant reports record quarterly earnings beating analyst expectations",
        "new artificial intelligence chip promises tenfold performance improvement",
        "software company acquires startup for billion dollar deal",
        "semiconductor shortage continues to impact global supply chain",
        "cloud computing revenues surge as enterprises accelerate digital transformation",
        "cybersecurity breach exposes millions of user records at major firm",
        "electric vehicle battery technology achieves breakthrough energy density",
        "social media platform announces major algorithm update for content ranking",
    ],
    "finance": [
        "stock market rallies as inflation data shows signs of cooling",
        "central bank holds interest rates steady amid economic uncertainty",
        "hedge fund reports massive gains on short positions this quarter",
        "cryptocurrency markets tumble as regulatory concerns mount globally",
        "merger and acquisition activity reaches decade high in corporate sector",
        "bond yields surge following stronger than expected employment report",
        "investment bank upgrades equities forecast citing improving fundamentals",
        "private equity firm closes largest buyout fund in firm history",
    ],
    "trade": [
        "new tariffs on imported goods expected to raise consumer prices",
        "trade deficit narrows as export demand surges from emerging markets",
        "shipping container shortage disrupts global supply chains worldwide",
        "free trade agreement between nations enters final stages of negotiation",
        "port congestion causes weeks long delays for imported merchandise",
        "currency devaluation boosts export competitiveness for domestic producers",
        "commodity prices spike following supply disruptions in key producing regions",
        "import restrictions on critical minerals raise national security concerns",
    ],
    "health": [
        "clinical trial results show promising outcomes for new cancer treatment",
        "pharmaceutical company receives regulatory approval for blockbuster drug",
        "global health organization warns of emerging infectious disease threat",
        "hospital networks report record patient volumes straining resources",
        "medical device breakthrough enables minimally invasive surgical procedures",
        "mental health crisis among young adults prompts policy response",
        "vaccine rollout accelerates as manufacturing capacity expands rapidly",
        "biotech startup raises funding for gene therapy research program",
    ],
    "sports": [
        "championship team celebrates historic victory after dramatic final match",
        "star athlete signs record breaking contract extension with franchise",
        "international tournament breaks viewership records across streaming platforms",
        "coaching staff overhaul signals major shift in team strategy",
        "young prospect earns call up following outstanding minor league season",
        "sports federation announces updated doping policy and testing protocols",
        "venue expansion project approved to accommodate growing fan demand",
        "athlete foundation launches program supporting underprivileged youth athletes",
    ],
}

TITANIC_TEMPLATES = {
    1: [  # survived
        "first class passenger woman rescued safely from lifeboat",
        "young child saved by crew member during evacuation",
        "female passenger from wealthy family among survivors",
        "officer assisted passengers into lifeboat and survived",
        "woman traveling alone rescued from freezing waters",
    ],
    0: [  # did not survive
        "third class male passenger lost at sea in disaster",
        "crew member who stayed to help others did not survive",
        "young man traveling to start new life perished in sinking",
        "family separated during evacuation father did not make it",
        "male passenger in second class among those who perished",
    ],
}


def make_tweet(label: int) -> str:
    if label == 1:
        tmpl = random.choice(POS_TWEET_TEMPLATES)
        adj = random.choice(POS_ADJ)
    else:
        tmpl = random.choice(NEG_TWEET_TEMPLATES)
        adj = random.choice(NEG_ADJ)
    return tmpl.format(noun=random.choice(NOUNS), adj=adj)


def make_review(label: int) -> str:
    base = random.choice(POS_REVIEWS if label == 1 else NEG_REVIEWS)
    # Add some variation
    extras = [
        "the cinematography was stunning", "the acting was top notch",
        "the pacing felt off at times", "the soundtrack was memorable",
        "the ending was surprising", "the characters were well developed",
        "the dialogue felt natural", "the special effects were impressive",
    ]
    n_extra = random.randint(0, 2)
    parts = [base] + random.sample(extras, n_extra)
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TWEETS DATASET (train = 100k, test1 = 5k)
# ─────────────────────────────────────────────────────────────────────────────
def generate_tweets(n: int, path: str):
    rows = []
    for _ in range(n):
        label = random.randint(0, 1)
        rows.append({"text": make_tweet(label), "label": label, "source": "tweets"})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved {n:,} tweets → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOVIE REVIEWS DATASET (train = 40k, test2 part)
# ─────────────────────────────────────────────────────────────────────────────
def generate_movies(n: int, path: str):
    rows = []
    for _ in range(n):
        label = random.randint(0, 1)
        rows.append({"text": make_review(label), "label": label, "source": "movies"})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved {n:,} movie reviews → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRADE / NEWS HEADLINES (multi-class)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP_TRADE = {c: i for i, c in enumerate(sorted(TRADE_CLASSES))}


def generate_trade(n: int, path: str):
    rows = []
    classes = list(TRADE_CLASSES.keys())
    for _ in range(n):
        cls = random.choice(classes)
        headline = random.choice(TRADE_CLASSES[cls])
        # small perturbations
        words = headline.split()
        if random.random() < 0.3 and len(words) > 5:
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(words)
        rows.append({
            "text": " ".join(words),
            "label": LABEL_MAP_TRADE[cls],
            "label_name": cls,
            "source": "trade_news",
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved {n:,} trade headlines → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TITANIC TEXT (binary)
# ─────────────────────────────────────────────────────────────────────────────
def generate_titanic(n: int, path: str):
    rows = []
    for _ in range(n):
        label = random.choice([0, 1])
        text = random.choice(TITANIC_TEMPLATES[label])
        rows.append({"text": text, "label": label, "source": "titanic"})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved {n:,} titanic records → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED TRAINING SET
# ─────────────────────────────────────────────────────────────────────────────
def generate_all():
    print("Generating datasets…")

    # Training data
    tweets_train = generate_tweets(80_000, f"{DATA_DIR}/tweets_train.csv")
    movies_train = generate_movies(30_000, f"{DATA_DIR}/movies_train.csv")
    titanic_train = generate_titanic(5_000, f"{DATA_DIR}/titanic_train.csv")
    trade_train = generate_trade(20_000, f"{DATA_DIR}/trade_train.csv")

    # Combined binary train (tweets + movies + titanic)
    combined = pd.concat([
        tweets_train[["text", "label"]],
        movies_train[["text", "label"]],
        titanic_train[["text", "label"]],
    ], ignore_index=True).sample(frac=1, random_state=42)
    combined.to_csv(f"{DATA_DIR}/train_primary.csv", index=False)
    print(f"  Combined primary train: {len(combined):,} rows")

    # Test Dataset 1 – tweets (unseen)
    tweets_test = generate_tweets(5_000, f"{DATA_DIR}/test_dataset1_tweets.csv")

    # Test Dataset 2 – mixed (movies + trade)
    movies_test = generate_movies(3_000, f"{DATA_DIR}/movies_test.csv")
    trade_test = generate_trade(3_000, f"{DATA_DIR}/trade_test.csv")
    mixed_test = pd.concat([
        movies_test[["text", "label"]],
        trade_test[["text", "label"]],
    ], ignore_index=True).sample(frac=1, random_state=99)
    mixed_test.to_csv(f"{DATA_DIR}/test_dataset2_mixed.csv", index=False)
    print(f"  Test Dataset 2 (mixed): {len(mixed_test):,} rows")

    # Save label map for trade
    import json
    with open(f"{DATA_DIR}/trade_label_map.json", "w") as f:
        json.dump(LABEL_MAP_TRADE, f)

    print("\n✓ All datasets generated successfully.")
    return combined


if __name__ == "__main__":
    generate_all()
