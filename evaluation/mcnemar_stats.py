"""Compute McNemar test statistics from results.json."""
import json
from collections import Counter

r = json.load(open('evaluation/results.json', encoding='utf-8'))
details = r['details']

def mcnemar(subset):
    b = sum(1 for x in subset if x['mm_correct'] and not x['to_correct'])
    c = sum(1 for x in subset if not x['mm_correct'] and x['to_correct'])
    n = b + c
    chi2 = (abs(b - c) - 1)**2 / n if n > 0 else 0
    # p-value approximation
    import math
    def chi2_sf(x, df=1):
        # incomplete gamma via series for df=1 (chi2 = normal^2)
        return 1 - math.erf(math.sqrt(x/2))
    p = chi2_sf(chi2) if n > 0 else 1.0
    return b, c, round(chi2, 3), round(p, 4)

all_q    = details
visual   = [x for x in details if x['requires_visual']]
nonvis   = [x for x in details if not x['requires_visual']]

print('McNemar (MM vs TO Grounded):')
print('  %-20s  b=%3d  c=%3d  chi2=%6.3f  p=%.4f' % ('All (n=%d)'%len(all_q),   *mcnemar(all_q)))
print('  %-20s  b=%3d  c=%3d  chi2=%6.3f  p=%.4f' % ('Visual (n=%d)'%len(visual), *mcnemar(visual)))
print('  %-20s  b=%3d  c=%3d  chi2=%6.3f  p=%.4f' % ('NonVisual (n=%d)'%len(nonvis), *mcnemar(nonvis)))

# TO failure modes on visual questions
print()
print('TO Grounded failures on visual questions (%d total):' % len(visual))
refused = sum(1 for x in visual if not x['to_correct'] and '无法回答' in x['text_only_answer'])
wrong   = sum(1 for x in visual if not x['to_correct'] and '无法回答' not in x['text_only_answer'])
correct = sum(1 for x in visual if x['to_correct'])
total_fail = len(visual) - correct
print('  Correct:  %d (%.0f%%)' % (correct, correct/len(visual)*100))
print('  Refused:  %d (%.0f%% of failures)' % (refused, refused/max(total_fail,1)*100))
print('  Wrong:    %d (%.0f%% of failures)' % (wrong,   wrong/max(total_fail,1)*100))
