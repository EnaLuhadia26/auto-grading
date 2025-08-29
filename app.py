from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Example question bank with correct answers
qa_bank = {
     # Python
    "What is Python?": "Python is a high-level, interpreted programming language that is easy to learn and widely used for web development, data science, and scripting.",
    "Explain Python lists.": "A list in Python is a collection of items that is ordered, changeable, and allows duplicate elements.",
    "What is a Python dictionary?": "A dictionary in Python is an unordered collection of key-value pairs, where each key is unique.",
    "Explain Python functions.": "A function is a block of reusable code that performs a specific task, defined using the 'def' keyword.",
    "What is a Python tuple?": "A tuple is an ordered, immutable collection of items in Python.",
    "Explain Python sets.": "A set is an unordered collection of unique elements in Python.",
    "What is Python slicing?": "Slicing is a method to extract a subset of elements from a sequence using indices.",
    "Explain Python classes.": "Classes in Python are blueprints for creating objects, encapsulating data and methods.",

    # Machine Learning
    "Define machine learning.": "Machine learning is a field of artificial intelligence that uses algorithms to allow computers to learn from data and make predictions or decisions without being explicitly programmed.",
    "What is supervised learning?": "Supervised learning is a type of machine learning where the model is trained on labeled data to predict outputs.",
    "What is unsupervised learning?": "Unsupervised learning is a type of machine learning where the model finds patterns or structure in unlabeled data.",
    "Explain overfitting.": "Overfitting occurs when a machine learning model performs well on training data but poorly on unseen data.",
    "What is a decision tree?": "A decision tree is a flowchart-like model used for classification and regression, splitting data based on feature values.",
    "What is a random forest?": "Random forest is an ensemble method using multiple decision trees to improve accuracy and reduce overfitting.",
    "Explain linear regression.": "Linear regression predicts a continuous target variable using a linear combination of input features.",
    "Explain logistic regression.": "Logistic regression predicts a categorical outcome using a logistic function to model probabilities.",

    # Web Development
    "What is HTML?": "HTML stands for HyperText Markup Language and is used to create and structure content on the web.",
    "What is CSS?": "CSS stands for Cascading Style Sheets and is used to style HTML elements with colors, layouts, and fonts.",
    "What is JavaScript?": "JavaScript is a scripting language used to make web pages interactive and dynamic.",
    "Explain the difference between GET and POST requests.": "GET requests retrieve data from the server and are visible in the URL, while POST requests send data to the server and are not visible in the URL.",
    "What is the purpose of a CSS class?": "A CSS class allows multiple HTML elements to share the same styling rules.",
    "What is the difference between inline, internal, and external CSS?": "Inline CSS is written inside HTML tags, internal CSS is written in a <style> tag within HTML, and external CSS is in a separate file linked to HTML.",
    "What is the DOM?": "The DOM (Document Object Model) is a programming interface representing the HTML structure as objects.",
    "What is responsive web design?": "Responsive web design ensures web pages render well on different devices and screen sizes.",

    # Databases
    "What is SQL?": "SQL stands for Structured Query Language and is used to manage and manipulate relational databases.",
    "Explain primary key.": "A primary key is a unique identifier for a record in a database table.",
    "What is a foreign key?": "A foreign key is a field in one table that refers to the primary key in another table to establish a relationship.",
    "What is normalization?": "Normalization is the process of organizing database tables to reduce redundancy and improve data integrity.",
    "What is a JOIN in SQL?": "A JOIN combines rows from two or more tables based on a related column between them.",
    "What is an index in databases?": "An index improves database query performance by allowing faster data retrieval.",
    "Explain ACID properties.": "ACID stands for Atomicity, Consistency, Isolation, Durability, which ensure reliable database transactions.",

    # Data Structures
    "What is a stack?": "A stack is a linear data structure that follows the Last In First Out (LIFO) principle.",
    "What is a queue?": "A queue is a linear data structure that follows the First In First Out (FIFO) principle.",
    "What is a linked list?": "A linked list is a linear data structure where each element points to the next element in the sequence.",
    "What is a binary tree?": "A binary tree is a tree data structure where each node has at most two children.",
    "What is a hash table?": "A hash table stores key-value pairs and allows fast lookup using a hash function.",
    "Explain graph data structure.": "A graph is a collection of nodes (vertices) connected by edges, used to model relationships.",

    # Algorithms
    "What is an algorithm?": "An algorithm is a step-by-step procedure to solve a problem or perform a task.",
    "Explain recursion.": "Recursion is a process where a function calls itself to solve smaller instances of a problem.",
    "What is binary search?": "Binary search is an efficient algorithm to find an element in a sorted array by repeatedly dividing the search interval in half.",
    "What is merge sort?": "Merge sort is a divide-and-conquer algorithm that splits the array, sorts the parts, and merges them.",
    "What is time complexity?": "Time complexity measures the amount of time an algorithm takes to run relative to the input size.",

    # Networking
    "What is an IP address?": "An IP address is a unique identifier for a device on a network.",
    "What is TCP/IP?": "TCP/IP is a set of protocols used to connect devices on the internet and ensure reliable data transmission.",
    "What is DNS?": "DNS (Domain Name System) translates human-readable domain names to IP addresses.",
    "What is HTTP?": "HTTP (HyperText Transfer Protocol) is used for communication between web browsers and servers.",
    "What is HTTPS?": "HTTPS is the secure version of HTTP, encrypting data transmitted between client and server.",

    # AI
    "What is artificial intelligence?": "Artificial intelligence is the simulation of human intelligence in machines programmed to think and learn.",
    "What is supervised learning in AI?": "Supervised learning in AI is training a model on labeled data to make predictions.",
    "What is reinforcement learning?": "Reinforcement learning is a type of machine learning where agents learn to make decisions by receiving rewards or penalties.",
    "What is natural language processing?": "NLP is a field of AI that focuses on the interaction between computers and human language.",
    "What is computer vision?": "Computer vision is an AI field that enables machines to interpret and process visual information from the world.",

    # Object-Oriented Programming (OOP)
    "Explain OOP.": "Object-Oriented Programming (OOP) is a programming paradigm based on objects containing data and methods to operate on that data.",
    "What is inheritance in OOP?": "Inheritance allows a class to acquire properties and methods from another class, promoting code reuse.",
    "Explain polymorphism in OOP.": "Polymorphism is the ability of different objects to be treated as instances of the same class through a common interface.",
    "What is encapsulation?": "Encapsulation is the bundling of data and methods within a class, restricting direct access to some of the object's components."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')

    if not question or not answer:
        return jsonify({"score": 0, "feedback": "Please provide both question and answer."})

    if question not in qa_bank:
        return jsonify({"score": 0, "feedback": "Question not found in database."})

    correct_answer = qa_bank[question]

    # Calculate similarity
    vectorizer = TfidfVectorizer().fit([correct_answer, answer])
    vectors = vectorizer.transform([correct_answer, answer])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Convert similarity to 0-10 score
    score = round(similarity * 10, 1)

    # Generate feedback
    if score >= 8:
        feedback = "Excellent! Very close to the correct answer."
    elif score >= 5:
        feedback = "Good attempt, but missing some details."
    else:
        feedback = "Needs improvement. Check the key points in the answer."

    return jsonify({"score": score, "feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)




