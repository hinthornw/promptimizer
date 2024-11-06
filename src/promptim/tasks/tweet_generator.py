from promptim.trainer import PromptWrapper, Task


def under_180_chars(run, example):
    """Evaluate if the tweet is under 180 characters."""
    result = run.outputs.get("tweet", "")
    score = int(len(result) < 180)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "under_180_chars",
        "score": score,
        "comment": comment,
    }


def no_hashtags(run, example):
    """Evaluate if the tweet contains no hashtags."""
    result = run.outputs.get("tweet", "")
    score = int("#" not in result)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "no_hashtags",
        "score": score,
        "comment": comment,
    }


def multiple_lines(run, example):
    """Evaluate if the tweet contains multiple lines."""
    result = run.outputs.get("tweet", "")
    score = int("\n" in result)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "multiline",
        "score": score,
        "comment": comment,
    }


tweet_task = Task(
    name="Tweet Generator",
    dataset="tweet-optim",
    initial_prompt=PromptWrapper(identifier="tweet-generator-example:c39837bd"),
    evaluators=[under_180_chars, no_hashtags, multiple_lines],
    evaluator_descriptions={
        "under_180_chars": "Checks if the tweet is under 180 characters. 1 if true, 0 if false.",
        "no_hashtags": "Checks if the tweet contains no hashtags. 1 if true, 0 if false.",
        "multiline": "Fails if the tweet is not multiple lines. 1 if true, 0 if false. 0 is bad.",
    },
)


## Example of how to create the dataset

if __name__ == "__main__":
    from langsmith import Client

    client = Client()

    topics = [
        "NBA",
        "NFL",
        "Movies",
        "Taylor Swift",
        "Artificial Intelligence",
        "Climate Change",
        "Space Exploration",
        "Cryptocurrency",
        "Healthy Living",
        "Travel Destinations",
        "Technology Trends",
        "Fashion",
        "Food and Cooking",
        "Music Festivals",
        "Entrepreneurship",
        "Fitness",
        "Gaming",
        "Politics",
        "Environmental Conservation",
        "Social Media Trends",
        "Education",
        "Mental Health",
        "Renewable Energy",
        "Virtual Reality",
        "Sustainable Fashion",
        "Robotics",
        "Quantum Computing",
        "Genetic Engineering",
        "Smart Cities",
        "Cybersecurity",
        "Augmented Reality",
        "Electric Vehicles",
        "Blockchain",
        "3D Printing",
        "Nanotechnology",
        "Biotechnology",
        "Internet of Things",
        "Cloud Computing",
        "Big Data",
        "Machine Learning",
        "Artificial General Intelligence",
        "Space Tourism",
        "Autonomous Vehicles",
        "Drones",
        "Wearable Technology",
        "Personalized Medicine",
        "Telemedicine",
        "Remote Work",
        "Digital Nomads",
        "Gig Economy",
        "Circular Economy",
        "Vertical Farming",
        "Lab-grown Meat",
        "Plant-based Diets",
        "Mindfulness",
        "Yoga",
        "Meditation",
        "Biohacking",
        "Nootropics",
        "Intermittent Fasting",
        "HIIT Workouts",
        "Esports",
        "Streaming Services",
        "Podcasting",
        "True Crime",
        "Tiny Houses",
        "Minimalism",
        "Zero Waste Living",
        "Upcycling",
        "Eco-tourism",
        "Voluntourism",
        "Digital Detox",
        "Slow Living",
        "Hygge",
        "Urban Gardening",
        "Permaculture",
        "Regenerative Agriculture",
        "Microplastics",
        "Ocean Conservation",
        "Rewilding",
        "Endangered Species",
        "Biodiversity",
        "Ethical AI",
        "Data Privacy",
        "Net Neutrality",
        "Deepfakes",
        "Fake News",
        "Social Media Activism",
        "Cancel Culture",
        "Meme Culture",
        "NFTs",
        "Decentralized Finance",
        "Universal Basic Income",
        "Gender Equality",
        "LGBTQ+ Rights",
        "Black Lives Matter",
        "Indigenous Rights",
        "Refugee Crisis",
        "Global Poverty",
        "Universal Healthcare",
        "Drug Decriminalization",
        "Prison Reform",
        "Gun Control",
        "Voting Rights",
        "Gerrymandering",
        "Campaign Finance Reform",
        "Term Limits",
        "Ranked Choice Voting",
        "Direct Democracy",
        "Space Debris",
        "Asteroid Mining",
        "Mars Colonization",
        "Extraterrestrial Life",
        "Dark Matter",
        "Black Holes",
        "Quantum Entanglement",
        "Fusion Energy",
        "Antimatter",
        "Cryonics",
        "Life Extension",
        "Transhumanism",
        "Cyborgs",
        "Brain-Computer Interfaces",
        "Memory Implants",
        "Holographic Displays",
    ]

    # Create datasets
    ds = client.create_dataset(dataset_name="tweet-optim")

    # Split topics into train, dev, and test sets
    train_topics = topics[:80]
    dev_topics = topics[80:90]
    test_topics = topics[90:]

    # Create examples for each dataset
    for split_name, dataset_topics in [
        ("train", train_topics),
        ("dev", dev_topics),
        ("test", test_topics),
    ]:
        client.create_examples(
            inputs=[{"topic": topic} for topic in dataset_topics],
            dataset_id=ds.id,
            splits=[split_name] * len(dataset_topics),
        )

    print("Dataset created successfully!")
