# Mercor-ML-Task
 ```
 A Python-based tool that, when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile
```
>Light weight version of this project -> [GitHub Automated Analysis Tool](https://github.com/AbhishekRP2002/Github-Automated-Analysis-Tool)

## Getting Started

These instructions will help you set up the project and run it on your local machine.

### Prerequisites
```
- Install [Python](https://www.python.org/downloads/) 3.9.0 or later
- Set up a virtual environment if you want (Recommended)
```

### Installing

1. Clone the repository to your local machine.
   ```
   git clone https://github.com/AbhishekRP2002/Github-Automated-Analysis-Tool.git
   ```

2. Go to the project folder.
   ```
   cd Github-Automated-Analysis-Tool
   ```

3. Create a virtual environment.
   ```
   python -m venv venv
   ```

4. Activate the virtual environment.
   - On Windows:
       ```
       .\venv\Scripts\activate
       ```
   - On Linux or MacOS:
       ```
       source venv/bin/activate
       ```
> You can use `conda` or  packages for [setting the virtual environment](https://www.scaler.com/topics/how-to-create-requirements-txt-python/).
5. Install the required dependencies using the following command.
   ```
   pip install -r requirements.txt
   ```

## Running the application

1. Run the streamlit application.
   ```
   streamlit run app.py
   ```

2. Open your web browser and enter the URL shown in the terminal, usually `http://localhost:8501`

3. Enjoy and tweak your Python Streamlit project!

## Built With

- [Python](https://www.python.org/)
- [Streamlit](https://www.streamlit.io/)
- [LangChain](https://langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/introduction)
- [Activeloop.ai](https://www.activeloop.ai/)


## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.


