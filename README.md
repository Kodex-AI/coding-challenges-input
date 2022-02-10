**Static code analysis**

Imagine our client has provided us with a github repository containing the source code of his AI systems.

The challenge is to perform some very basic code analysis on a couple of python files provided by the client.

This means in some way letting the user start the process, accessing the github files, running some analysis on the source files, and then outputting the result in some meaningful way (could be a simple UI, otherwise command line is fine).

The analysis should count the number of hidden layers in the neural network architecture, and also count how many layers have a 'relu' activation function vs. a 'sigmoid' activation function.

This is e.g. how the neural network is initialized: `model = Sequential()`  
And this is e.g. how a hidden layer is added: `model.add(Dense(8, activation='relu'))`

The models should be classified as follows:
* 1-9 hidden layers: Low transparency risk
* 10-19 hidden layers: Medium transparency risk
* 20+ hidden layers: High transparency risk

Link to the client's github repository: https://github.com/Kodex-AI/coding-challenges-input


**Example outputs:**

"Found deep neural network with 21 hidden layers in dummy_ai_01.py (4 sigmoid activation functions, 17 relu activation functions), posing a high transparency risk."

"No hidden layers found in dummy_ai_04.py"

**Expected deliverables from your side**
* python code solving the challenge
* short README file giving instructions on how to use your code

**Hints:**
* You will have to find and apply the github API reference. (Don't just download the files.)
* For the sake of this challenge, the github repository has been made public, so no access token etc. is required.
* Unfortunately, coding standards aren't very high at our client's organization, so the syntax got a bit messed up in some of the provided files, which you will need to account for.
* Don't hesitate to contact me in case anything is unclear.