import rag_core


def chat_with_pdf():
    print("Initializing PDF chat system...")
    rag_core.build_index()
    session_id = "cli"
    rag_core.reset_session(session_id)

    print("\nChat initialized! Type 'quit' to exit.")
    print("Ask questions about your PDF:\n")

    while True:
        question = input("\nYou: ")

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if question.strip() == "":
            continue

        try:
            answer = rag_core.chat(session_id, question)
            print("\nAssistant:", answer)
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking your question again.")


if __name__ == "__main__":
    chat_with_pdf()
