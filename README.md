# CSE 151B Competition — Team O(Nxiety^1.5)
**By Mallika Dasgupta, Pranav Reddy Bussanagari, Aryen Singhal**

## Environment
- GPU used: A100 80GB on Google Colab Pro
- Total generation/inference time: 4-5 hours for 943 questions

## Model Setup
- We used the standard `Qwen/Qwen3-4B-Thinking-2507` model from Hugging Face.
- `run_inference()` automatically loads the model, no additional setup is required.

## Run Inference
1. Ensure the required libraries from `requirements.txt` are installed.
2. Ensure the environment uses cuda version >= 13.0.
3. The private dataset should be in the same directory as `run_inference.py` and named `private.jsonl`.
4. Execute the `run_inference.py` script, which contains a single function `run_inference()`.
5. The results will be saved in `submission.csv` in the same directory.
6. The data and output paths can be changed in the configuration section of `run_inference.py` if needed.
