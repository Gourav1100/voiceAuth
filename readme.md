# Voice Auth - Python
## Installation -
1. Create and activate virtual env -
   ```
   sudo apt install python3-venv
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install python packages
   ```
   pip install -r requirements.txt
   ```
3. Install system packages
   ```
   sudo apt install ffmpeg
   ----  OR ----
   brew install ffmpeg
   ```
## How to use -
1. Install all dependencies.
2. Run the following command -
    ```
    python3 -m venv venv
    source venv/bin/activate
    python main.py <username>
    ```