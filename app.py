from rag_pipeline import create_vectorstore, search


def main():
    vectorstore = create_vectorstore()

    while True:
        query = input("\nQuestion: ")
        if query == "exit":
            break

        results = search(query, vectorstore)

        print("\n🔎 Contexte trouvé:")
        for r in results:
            print("-", r[:200])


if __name__ == "__main__":
    main()
