"""Diagnostic-only analysis of aggregate total-goal bias bootstrap behavior."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values); position = (len(ordered)-1)*quantile; lower = int(position); fraction = position-lower
    return ordered[lower] if lower+1 == len(ordered) else ordered[lower]*(1-fraction)+ordered[lower+1]*fraction


def distribution_summary(values: list[float]) -> dict[str, Any]:
    mean=statistics.mean(values); std=statistics.pstdev(values)
    skewness=statistics.mean(((value-mean)/std)**3 for value in values) if std else 0.0
    return {"mean":mean,"median":statistics.median(values),"standard_deviation":std,"skewness":skewness,"quantiles":{str(q):percentile(values,q) for q in (.01,.025,.05,.1,.25,.5,.75,.9,.95,.975,.99)},"proportion_above_zero":sum(v>0 for v in values)/len(values),"proportion_above_0_05":sum(v>.05 for v in values)/len(values),"proportion_above_0_10":sum(v>.10 for v in values)/len(values),"proportion_above_0_20":sum(v>.20 for v in values)/len(values)}


def full_date_block_bootstrap(rows, production_predictions, candidate_predictions, *, samples: int, seed: int) -> dict[str, Any]:
    blocks: dict[str,list[int]]=defaultdict(list)
    for index,row in enumerate(rows): blocks[row.played_on.isoformat()].append(index)
    dates=sorted(blocks); rng=random.Random(seed); records=[]; upper_counts=Counter(); all_counts=Counter()
    for sample in range(samples):
        selected=[rng.choice(dates) for _ in dates]; indices=[i for day in selected for i in blocks[day]]
        actual=sum(rows[i].backtest.home_score+rows[i].backtest.away_score for i in indices)/len(indices)
        production=sum(sum(production_predictions[i][0]) for i in indices)/len(indices)-actual
        candidate=sum(sum(candidate_predictions[i][0]) for i in indices)/len(indices)-actual
        statistic=abs(candidate)-abs(production)
        records.append({"sample":sample,"production_signed_bias":production,"candidate_signed_bias":candidate,"candidate_minus_production_signed_bias":candidate-production,"absolute_bias_difference":statistic})
        counts=Counter(selected); all_counts.update(counts)
        if statistic>.10: upper_counts.update(counts)
    statistic=[r["absolute_bias_difference"] for r in records]; production=[r["production_signed_bias"] for r in records]; candidate=[r["candidate_signed_bias"] for r in records]
    upper_samples=sum(r["absolute_bias_difference"]>.10 for r in records)
    tail=[]
    for day in dates:
        overall=all_counts[day]/samples; upper=upper_counts[day]/upper_samples if upper_samples else 0
        tail.append({"date":day,"mean_draws_all_samples":overall,"mean_draws_upper_tail":upper,"upper_tail_enrichment":upper-overall,"matches_in_block":len(blocks[day])})
    return {"method":"paired date-block bootstrap with replacement","samples":samples,"seed":seed,"full_distribution":records,"absolute_bias_difference_summary":distribution_summary(statistic),"production_signed_bias_summary":distribution_summary(production),"candidate_signed_bias_summary":distribution_summary(candidate),"production_crosses_zero_frequency":sum(v<=0 for v in production)/samples,"candidate_crosses_zero_frequency":sum(v>=0 for v in candidate)/samples,"opposite_sign_frequency":sum((p<0<c or c<0<p) for p,c in zip(production,candidate))/samples,"upper_tail_block_enrichment":sorted(tail,key=lambda row:abs(row["upper_tail_enrichment"]),reverse=True)[:25]}


def calibration_line(predicted: list[float], actual: list[int]) -> dict[str,float]:
    mean_x=statistics.mean(predicted); mean_y=statistics.mean(actual); variance=sum((x-mean_x)**2 for x in predicted)
    slope=sum((x-mean_x)*(y-mean_y) for x,y in zip(predicted,actual))/variance if variance else 0.0
    return {"intercept":mean_y-slope*mean_x,"slope":slope,"mean_predicted":mean_x,"mean_actual":mean_y}


def reliability(predicted: list[float], actual: list[int]) -> list[dict[str,Any]]:
    bounds=((0,1.5),(1.5,2),(2,2.5),(2.5,3),(3,3.5),(3.5,float("inf"))); output=[]
    for lower,upper in bounds:
        selected=[(p,a) for p,a in zip(predicted,actual) if lower<=p<upper]
        output.append({"lower":lower,"upper":None if math.isinf(upper) else upper,"count":len(selected),"mean_predicted":statistics.mean(p for p,_ in selected) if selected else None,"mean_actual":statistics.mean(a for _,a in selected) if selected else None,"signed_gap":statistics.mean(p-a for p,a in selected) if selected else None})
    return output


def poisson_goal_bands(expected_totals: list[float], actual_totals: list[int]) -> dict[str,Any]:
    bands=("0_1","2","3","4","5_plus"); predicted={key:0.0 for key in bands}; observed={key:0 for key in bands}
    for expected,actual in zip(expected_totals,actual_totals):
        probabilities=[math.exp(-expected)*expected**goal/math.factorial(goal) for goal in range(5)]
        predicted["0_1"]+=probabilities[0]+probabilities[1]; predicted["2"]+=probabilities[2]; predicted["3"]+=probabilities[3]; predicted["4"]+=probabilities[4]; predicted["5_plus"]+=1-sum(probabilities)
        key="0_1" if actual<=1 else str(actual) if actual<=4 else "5_plus"; observed[key]+=1
    n=len(actual_totals); return {"predicted_rates":{k:v/n for k,v in predicted.items()},"observed_rates":{k:v/n for k,v in observed.items()}}


def block_error_diagnostics(rows, production_predictions, candidate_predictions) -> dict[str,Any]:
    blocks: dict[str,list[int]]=defaultdict(list)
    for i,row in enumerate(rows): blocks[row.played_on.isoformat()].append(i)
    details=[]
    for day,indices in sorted(blocks.items()):
        actual=statistics.mean(rows[i].backtest.home_score+rows[i].backtest.away_score for i in indices)
        production=statistics.mean(sum(production_predictions[i][0]) for i in indices)-actual; candidate=statistics.mean(sum(candidate_predictions[i][0]) for i in indices)-actual
        details.append({"date":day,"matches":len(indices),"actual_mean":actual,"production_bias":production,"candidate_bias":candidate,"absolute_bias_difference":abs(candidate)-abs(production)})
    return {"blocks":details,"mean_absolute_block_bias":{"production":statistics.mean(abs(r["production_bias"]) for r in details),"candidate":statistics.mean(abs(r["candidate_bias"]) for r in details)},"root_mean_square_block_bias":{"production":math.sqrt(statistics.mean(r["production_bias"]**2 for r in details)),"candidate":math.sqrt(statistics.mean(r["candidate_bias"]**2 for r in details))},"largest_candidate_harm_blocks":sorted(details,key=lambda r:r["absolute_bias_difference"],reverse=True)[:25]}
