from utils import chat
import os

os.makedirs("outputres", exist_ok=True)


def client():
    messages = []
    print(
        "[ Bot ]: Waiting for a task... (input `exit` to exit)")
    while True:
        userinput = input("[ User ]: ")
        if userinput == "exit":
            break
        messages.append({"role": "user", "content": userinput})
        answer = chat(messages, userinput)
        print("[ Bot ]: ", answer)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    client()
