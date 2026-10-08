"""Published Massive-STEPS reference numbers (arXiv 2505.11239, Tables 4-5).
SUP  = best supervised model per city (STHGCN, in-city training)   (Acc@1, NDCG@5)
ZS   = best zero-shot LLM per city (max over the four LLM-Move rows, each metric separately).
       Protocol (Massive-STEPS src/run_next_poi_llm.py + prompts.py): 200 test users (seed 42),
       1 test trajectory each, last 15 TRAIN trajectories as history, candidates = ground truth
       + 100 POIs drawn UNIFORMLY AT RANDOM from the city, listed by distance -> our "gtrand100".
       Acc@1 checked against the repository README table (2026-10-07).
       NDCG@5 values: VERIFY against the paper table before submission.
ZS_GEMINI = LLM-Move with Gemini 2.0 Flash only (a fixed model, no per-city oracle choice)."""
SUP = {"bandung": (0.219, 0.302), "beijing": (0.453, 0.552), "istanbul": (0.241, 0.318),
       "jakarta": (0.197, 0.270), "kuwait_city": (0.225, 0.314), "melbourne": (0.168, 0.247),
       "moscow": (0.223, 0.308), "new_york": (0.146, 0.207), "palembang": (0.246, 0.341),
       "petaling_jaya": (0.174, 0.241), "sao_paulo": (0.250, 0.344), "shanghai": (0.193, 0.264),
       "sydney": (0.227, 0.307), "tangerang": (0.293, 0.400), "tokyo": (0.250, 0.350)}
ZS = {"bandung": (0.225, 0.289), "beijing": (0.192, 0.280), "istanbul": (0.205, 0.295),
      "jakarta": (0.295, 0.350), "kuwait_city": (0.220, 0.295), "melbourne": (0.225, 0.275),
      "moscow": (0.230, 0.330), "new_york": (0.235, 0.325), "palembang": (0.260, 0.329),
      "petaling_jaya": (0.210, 0.273), "sao_paulo": (0.285, 0.350), "shanghai": (0.170, 0.221),
      "sydney": (0.230, 0.331), "tangerang": (0.200, 0.275), "tokyo": (0.250, 0.368)}
ZS_GEMINI = {"bandung": 0.225, "beijing": 0.096, "istanbul": 0.205, "jakarta": 0.295,
             "kuwait_city": 0.220, "melbourne": 0.225, "moscow": 0.220, "new_york": 0.235,
             "palembang": 0.260, "petaling_jaya": 0.210, "sao_paulo": 0.285, "shanghai": 0.170,
             "sydney": 0.230, "tangerang": 0.200, "tokyo": 0.250}
