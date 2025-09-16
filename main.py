import re

spam = dict()
ham = dict()
total = 0

def tokenize(filename):
    global total

    try:
        file = open(filename, 'r')

        # Iterate over each message in dataset
        for line in file:
            # Remove all whitespace and make everything lowercase for consistency
            message = line.strip().lower().split("\t")
            # Remove all punctuation for consistency
            message[1] = re.sub(r'[^a-z0-9\s]', '', message[1])
            tokens = message[1].split(" ")

            # Iterate over each word in a message
            for i in range(1, len(tokens)):
                token = tokens[i]

                # Count the frequency of each word based on classification
                if (message[0] == "spam"):
                        spam[token] = spam.get(token, 0) + 1
                elif (message[0] == "ham"):
                        ham[token] = ham.get(token, 0) + 1

            total += len(tokens) - 1

    except FileNotFoundError:
        print("File not found.")

def main():
    filename = "SMSSpamCollection"
    tokenize(filename)

if __name__ == "__main__":
    main()