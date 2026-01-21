"""
Script for sending requests to the sentiment analysis API with gradually
more negative reviews over time (simulating data drift).
"""

import argparse
import random
import time

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8080/inference")
    parser.add_argument("--wait_time", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    reviews = [
        "The cinematography was breathtaking and the acting superb.",
        "A heartwarming story that kept me engaged till the end.",
        "Solid performance by the lead actor, though the plot was predictable.",
        "An absolute masterpiece of modern cinema.",
        "The visual effects were stunning, a true spectacle.",
        "A decent watch, good for a weekend movie night.",
    ]

    negative_phrases = [
        "However, the script was incredibly weak.",
        "The pacing was painfully slow.",
        "I was bored out of my mind halfway through.",
        "The dialogue felt forced and unnatural.",
        "It was a complete waste of time and money.",
    ]

    count = 0
    while count < args.max_iterations:
        review = random.choice(reviews)
        negativity_probability = min(count / args.max_iterations, 1.0)

        updated_review = review
        for phrase in negative_phrases:
            if random.random() < negativity_probability:
                updated_review += " " + phrase

        response = requests.post(args.url, json={"statement": updated_review}, timeout=10)
        print(f"Iteration {count}, Sent review: {updated_review}, Response: {response.json()}")
        time.sleep(args.wait_time)
        count += 1