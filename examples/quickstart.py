"""Standalone, explicitly synthetic semantic target search demo."""
import pandas as pd
from semanticavas import SemanticAtlas, synthesize

rows=[]
for participant in ("P1", "P2", "P3"):
    for stimulus, truck, annoying in (("A", 80., 20.), ("B", 45., 65.), ("C", 70., 30.)):
        rows.append({"participant":participant, "stimulus":stimulus, "truck":truck, "annoying":annoying})
ratings=pd.DataFrame(rows)
atlas=SemanticAtlas(ratings,["truck", "annoying"])
ranking=atlas.rank({"truck":80.,"annoying":20.},scale_ranges={"truck":100.,"annoying":100.})
assert ranking.index[0]=="A"
signal=synthesize(f0=80.,cutoff=2000.,duration=.2,seed=2026)
assert len(signal)>0
print("SemanticAVAS synthetic demo OK: winner A; WAV prototype array generated")
