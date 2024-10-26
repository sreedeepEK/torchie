## Torchie: The unofficial Pytorch Chatbot

### Description
Torchie is a tool designed to extract and retrieve information efficiently from PyTorch documentation. It aims to navigate the extensive PyTorch resources and provide quick access to relevant insights, making it easier for users to find the information they need without any hassle. [Click here to see the live demo.](https://huggingface.co/spaces/sreedeepEK/torchie)

### Project Status
This project is currently under development. Features and functionalities are being continuously added.

#### Planned Features and Functionalities

- **Keyword Search**
Allow users to search for specific keywords or phrases in the PyTorch documentation, providing quick access to relevant sections.

- **Code Examples**
Provide users with relevant code snippets and examples based on their queries, making it easier to understand how to implement specific functionalities in PyTorch.

- **FAQ Section**
Create a comprehensive FAQ section that addresses common questions and issues users face when using PyTorch.

- **Tutorial Recommendations**
Suggest tutorials and guides based on user queries, helping users learn specific aspects of PyTorch.

- **Integration with Other Libraries**
Provide information on how PyTorch integrates with other libraries like NumPy, SciPy, and TensorBoard, helping users leverage a broader ecosystem.

- **Version Compatibility Checker**
Enable users to check if certain functions or features are available in different versions of PyTorch, assisting in maintaining code compatibility.


### How to Run the Torchie App 

- Prerequisites

1. Python Installation: Ensure you have Python installed on your machine (preferably version 3.6 or later). You can download it from the official Python website.

2. Install Required Libraries: Torchie requires several Python libraries. You can install them using pip. Open your command line or terminal and run:

    ```
    pip install -r requirements.txt
    ```

- Clone the Repository

    ```bash

    git clone https://github.com/yourusername/torchie.git
    ```

- Navigate to the Project Directory:

    ```bash
    cd torchie
    ```

- Set Up Groq API

1. Obtain Your Groq API Key:
    Follow the steps mentioned  to sign up at Groq's official website and obtain your API key.

2. Store Your API Key:
    Create a configuration file (e.g., `config.py`) or use environment variables to securely store your Groq API key. Here’s an example of how to do this in a `.env` file:

    ```python

        GROQ_API_KEY = "YOUR_GROQ_API_KEY"  
    ``` 

- Run the Application

    Start the Gradio Server: Run the following command to start the server:

    ```bash

    python app.py
    ```

    If you are using Jupyter Notebook, you can also run:

    ```bash

    !python app.py
    ```

    Access the Application: Once the server is running, open your web browser and go to:

    ```
    http://127.0.0.1:5000
    ```

### License

Torchie has a MIT license, as found in [LICENSE](https://github.com/sreedeepEK/torchie/blob/main/LICENSE) file.
