from textbookapi import Principlesofdatascience, Introductiontopythonprogramming

API_KEY = "ujjwal-21754d5156915580033ec53670a12bb0580a8ccef50784ab"

print("=" * 50)
print("  Multi-Book RAG Chatbot")
print("  Type 'quit' to exit")
print("=" * 50)
print()
print("Choose a book:")
print("  1. Principles of Data Science")
print("  2. Introduction to Python Programming")
print()

choice = input("Enter 1 or 2: ").strip()
if choice == "2":
    client = Introductiontopythonprogramming(api_key=API_KEY)
    book_name = "Introduction to Python Programming"
else:
    client = Principlesofdatascience(api_key=API_KEY)
    book_name = "Principles of Data Science"

print()
print(f"  Chatting with: {book_name}")
print("=" * 50)

while True:
    print()
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    print("\nAssistant: ", end="")
    for token in client.ask(question, stream=True):
        print(token, end="", flush=True)
    print()
