"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for fake account / bot detection.
Mimics statistical distributions seen in real-world datasets like Cresci-2017
and TwiBot-20.

Classes:
  0 = Legitimate account
  1 = Bot / Fake account

Features (28 total):
  Profile-based, Social-graph, Activity/Behavioral, Content/NLP
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_LEGIT = 6000
N_BOT   = 4000
N_TOTAL = N_LEGIT + N_BOT


def generate_legit(n):
    d = {}
    # Profile
    d['account_age_days']     = np.random.normal(900, 400, n).astype(int).clip(30, 5000)
    d['has_profile_photo']    = np.random.choice([0, 1], n, p=[0.05, 0.95])
    d['has_bio']              = np.random.choice([0, 1], n, p=[0.15, 0.85])
    d['name_has_numbers']     = np.random.choice([0, 1], n, p=[0.75, 0.25])
    d['name_length']          = np.random.normal(10, 3, n).astype(int).clip(3, 25)
    d['name_entropy']         = np.random.normal(2.8, 0.5, n).clip(1.0, 4.5)
    d['is_verified']          = np.random.choice([0, 1], n, p=[0.97, 0.03])
    # Social graph
    d['followers_count']      = np.random.lognormal(5.5, 1.5, n).astype(int).clip(0, 500000)
    d['following_count']      = np.random.lognormal(4.8, 1.2, n).astype(int).clip(0, 5000)
    d['ff_ratio']             = np.clip(d['followers_count'] / (d['following_count'] + 1), 0, 200)
    d['mutual_follow_rate']   = np.random.beta(3, 2, n).clip(0.1, 1.0)
    d['listed_count']         = np.random.lognormal(2.0, 1.5, n).astype(int).clip(0, 5000)
    # Activity
    d['total_posts']          = np.random.lognormal(5.0, 1.5, n).astype(int).clip(1, 100000)
    d['avg_daily_posts']      = np.random.lognormal(0.5, 0.8, n).clip(0.01, 50)
    d['posting_hour_std']     = np.random.normal(6.0, 2.0, n).clip(0.5, 12)
    d['repost_ratio']         = np.random.beta(2, 5, n).clip(0, 1)
    d['reply_ratio']          = np.random.beta(3, 4, n).clip(0, 1)
    d['burst_score']          = np.random.beta(2, 6, n).clip(0, 1)
    d['inactive_days_ratio']  = np.random.beta(2, 4, n).clip(0, 1)
    # Content
    d['avg_hashtags_per_post']= np.random.lognormal(0.6, 0.7, n).clip(0, 30)
    d['avg_urls_per_post']    = np.random.beta(1.5, 6, n).clip(0, 5)
    d['duplicate_post_ratio'] = np.random.beta(1, 8, n).clip(0, 1)
    d['sentiment_variance']   = np.random.normal(0.6, 0.2, n).clip(0, 1)
    d['content_diversity']    = np.random.beta(4, 2, n).clip(0, 1)
    d['spam_keyword_score']   = np.random.beta(1, 9, n).clip(0, 1)
    d['mention_ratio']        = np.random.beta(2, 5, n).clip(0, 1)
    # Temporal derived
    d['account_age_activity_ratio'] = d['total_posts'] / (d['account_age_days'] + 1)
    d['followers_growth_rate']      = np.random.normal(0.5, 1.0, n).clip(-5, 50)
    df = pd.DataFrame(d)
    df['label'] = 0
    return df


def generate_bot(n):
    d = {}
    bot_type = np.random.choice(['spambot','fake_follower','social_bot','cyborg'], n,
                                 p=[0.35, 0.30, 0.25, 0.10])
    # Profile
    d['account_age_days']     = np.where(
        bot_type == 'cyborg',
        np.random.normal(600, 300, n).clip(30, 4000),
        np.random.normal(120, 90, n).clip(1, 800)
    ).astype(int)
    d['has_profile_photo']    = np.where(
        np.isin(bot_type, ['cyborg','social_bot']),
        np.random.choice([0,1], n, p=[0.2, 0.8]),
        np.random.choice([0,1], n, p=[0.55, 0.45])
    )
    d['has_bio']              = np.where(
        bot_type == 'cyborg',
        np.random.choice([0,1], n, p=[0.15, 0.85]),
        np.random.choice([0,1], n, p=[0.60, 0.40])
    )
    d['name_has_numbers']     = np.random.choice([0, 1], n, p=[0.25, 0.75])
    d['name_length']          = np.random.normal(14, 4, n).astype(int).clip(3, 25)
    d['name_entropy']         = np.random.normal(3.8, 0.5, n).clip(1.0, 4.5)
    d['is_verified']          = np.zeros(n, dtype=int)
    # Social graph
    d['following_count']      = np.random.lognormal(7.0, 1.0, n).astype(int).clip(100, 200000)
    d['followers_count']      = np.where(
        bot_type == 'fake_follower',
        np.random.lognormal(3.0, 1.0, n).astype(int).clip(0, 5000),
        np.random.lognormal(4.5, 1.5, n).astype(int).clip(0, 100000)
    )
    d['ff_ratio']             = np.clip(d['followers_count'] / (d['following_count'] + 1), 0, 10)
    d['mutual_follow_rate']   = np.random.beta(1, 5, n).clip(0, 0.3)
    d['listed_count']         = np.random.lognormal(0.5, 1.0, n).astype(int).clip(0, 500)
    # Activity
    d['total_posts']          = np.where(
        bot_type == 'fake_follower',
        np.random.lognormal(1.5, 1.0, n).astype(int).clip(0, 200),
        np.random.lognormal(5.5, 1.5, n).astype(int).clip(0, 200000)
    )
    d['avg_daily_posts']      = np.where(
        bot_type == 'spambot',
        np.random.lognormal(3.0, 0.8, n).clip(5, 500),
        np.random.lognormal(0.8, 1.0, n).clip(0.01, 100)
    )
    d['posting_hour_std']     = np.random.normal(2.5, 1.5, n).clip(0, 8)
    d['repost_ratio']         = np.random.beta(5, 2, n).clip(0, 1)
    d['reply_ratio']          = np.random.beta(1, 6, n).clip(0, 1)
    d['burst_score']          = np.random.beta(6, 2, n).clip(0, 1)
    d['inactive_days_ratio']  = np.random.beta(4, 2, n).clip(0, 1)
    # Content
    d['avg_hashtags_per_post']= np.random.lognormal(2.0, 0.8, n).clip(0, 50)
    d['avg_urls_per_post']    = np.random.beta(5, 3, n).clip(0, 10)
    d['duplicate_post_ratio'] = np.random.beta(5, 2, n).clip(0, 1)
    d['sentiment_variance']   = np.random.normal(0.2, 0.15, n).clip(0, 1)
    d['content_diversity']    = np.random.beta(1, 5, n).clip(0, 1)
    d['spam_keyword_score']   = np.random.beta(6, 2, n).clip(0, 1)
    d['mention_ratio']        = np.random.beta(5, 3, n).clip(0, 1)
    # Temporal derived
    d['account_age_activity_ratio'] = d['total_posts'] / (d['account_age_days'] + 1)
    d['followers_growth_rate']      = np.random.normal(5.0, 3.0, n).clip(-5, 100)
    df = pd.DataFrame(d)
    df['label'] = 1
    df['bot_type'] = bot_type
    return df


def generate_dataset():
    legit = generate_legit(N_LEGIT)
    bots  = generate_bot(N_BOT)
    bots_meta = bots[['bot_type']].copy()
    bots_clean = bots.drop(columns=['bot_type'])
    df = pd.concat([legit, bots_clean], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # light noise
    for col in ['avg_daily_posts', 'ff_ratio', 'burst_score', 'name_entropy']:
        df[col] += np.random.normal(0, df[col].std() * 0.05, len(df))
        df[col] = df[col].clip(0)
    return df, bots_meta


if __name__ == "__main__":
    df, bots_meta = generate_dataset()
    df.to_csv('/home/claude/bot_detection/dataset.csv', index=False)
    bots_meta.to_csv('/home/claude/bot_detection/bot_types.csv', index=False)
    print(f"Dataset: {df.shape}, Label dist:\n{df['label'].value_counts()}")
