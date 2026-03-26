from manager_agent import ManagerAgent

manager = ManagerAgent()

queries = [
    "How do I implement binary search in Python?"
]

for query in queries:

    response = manager.run(query)

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)
    print(f"Domain            : {response['domain']}")
    print(f"Domain Confidence : {response['domain_confidence']}")
    print(f"Agent Confidence  : {response['agent_confidence']}")

    if response['low_confidence']:
        print("⚠️  WARNING: Low confidence — answer may be unreliable")

    print(f"\nAnswer:\n{response['answer']}")
    print("=" * 60 + "\n")