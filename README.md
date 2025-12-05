# MILP-Alternatives-to-MaxSAT-for-Interpretable-Model-Synthesis
This project, completed for CS781: Formal Methods in ML (Fall 2025), explores MILP-based alternatives to MaxSAT for synthesizing interpretable decision diagrams. It reproduces the original encoding, analyzes its issues, and develops more efficient MILP/IP formulations under richer structural constraints.
Our  contribution to this can be found in FinalSubsubs
./FinalSubsubs contains our contribution to this prior work




# setup
use 
1. `pip install -U python-sat`
2. download and make bc2cnf from the original synplicate README
3. copy the bc2cnf binary to synplicate/synthesizer/max_sat
4. setup venv (use source venv/bin/activate) and install pandas, gurobipy
5. run `python3 synplicate.py experiments/ICML/AutoTaxi/`
6. install tensorflow 2.15 for the original datasets
7. pip install tabulate

8. CHKPNT1.notes = 
		"[done]Explainivility funcuton(34),
		SOTA MaxSAT,
		Continuos relaxation,
		Debias maxsat,
		try parent child constraint,
		[done, now maxexplainibility=maxcorrectness]sample weight normalisation,"
