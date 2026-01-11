# Alameda-vsli-placement
A machine learning system that learns to optimize chip layouts using differentiable loss functions. The model iteratively takes actions to reposition components, learning to eliminate overlaps and reduce wirelength through gradient-based feedback, helping automate a critical step in hardware design.

## How to Run / Implement the Project

Clone the repository:  
bash  
git clone https://github.com/manuhalapeth/Alameda-vsli-placement  
cd Alameda-vsli-placement  

Install dependencies (Python 3 required):  
pip install torch matplotlib  

Run the placement optimizer:  
python placement.py  

Run the test and evaluation script:  
python test.py  

