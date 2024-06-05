Eloquia

https://github.com/apple/ml-stuttering-events-dataset

https://www.kaggle.com/datasets/ikrbasak/sep-28k


Using Environment Variables
Set an Environment Variable:

On Windows:
Open Command Prompt and set the variable by typing: setx HF_TOKEN "your_hf_token_here". This sets the environment variable permanently.
Alternatively, you can set it for the current session only by using set HF_TOKEN="your_hf_token_here" in the command line.
On macOS/Linux:
Open Terminal and add the export command to your shell configuration file (.bashrc, .bash_profile, or .zshrc): echo 'export HF_TOKEN="your_hf_token_here"' >> ~/.bashrc (replace .bashrc with your configuration file as needed).
Reload the configuration file: source ~/.bashrc (replace with your specific file).
Environment variables can be accessed in Python using the os module:
python
Copy code
import os
hf_token = os.getenv('HF_TOKEN')
