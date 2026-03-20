from rag_pipeline import create_index, search

def main():
    index, chunks = create_index()

    while True:
        query = input("\nQuestion: ")
        if query == "exit":
            break

        results = search(query, index, chunks)

        print("\n🔎 Contexte trouvé:")
        for r in results:
            print("-", r[:200])

if __name__ == "__main__":
    main()