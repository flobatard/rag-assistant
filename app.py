from rag_pipeline import create_vector_db, create_qa_chain

def main():
    print("1. Indexer les documents")
    print("2. Poser une question")
    choice = input("Choix: ")

    if choice == "1":
        create_vector_db()

    elif choice == "2":
        qa = create_qa_chain()

        while True:
            query = input("\nQuestion: ")
            if query.lower() == "exit":
                break

            result = qa(query)

            print("\n🧠 Réponse:")
            print(result["result"])

            print("\n📚 Sources:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source"))

if __name__ == "__main__":
    main()