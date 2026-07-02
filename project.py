# Unique Python snippet: Auto-generate haiku poems
import random

# Word pools
nouns = ["clouds", "dreams", "code", "stars", "waves"]
verbs = ["flow", "rise", "fall", "glow", "drift"]
adjectives = ["silent", "bright", "endless", "fragile", "hidden"]

def haiku():
    line1 = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)}"
    line2 = f"{random.choice(nouns)} {random.choice(verbs)} {random.choice(adjectives)}"
    line3 = f"{random.choice(adjectives)} {random.choice(nouns)}"
    return "\n".join([line1, line2, line3])

# Generate 3 random haikus
for _ in range(3):
    print(haiku(), "\n")
