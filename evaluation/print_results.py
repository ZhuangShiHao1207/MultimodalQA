import json
r = json.load(open('evaluation/results.json', encoding='utf-8'))
o = r['overall']
print('=== OVERALL ===')
print('MM    ANLS=%.4f  Acc=%.2f%%' % (o['multimodal_anls'], o['multimodal_accuracy']*100))
print('TO    ANLS=%.4f  Acc=%.2f%%' % (o['text_only_anls'],  o['text_only_accuracy']*100))
print('Open  ANLS=%.4f  Acc=%.2f%%' % (o['text_open_anls'],  o['text_open_accuracy']*100))
print('MM vs TO:   +%.1f pp' % (o['mm_vs_to_improvement']*100))
print('MM vs Open: +%.1f pp' % (o['mm_vs_open_improvement']*100))
print('Total questions:', r['total_questions'])
print()
print('=== BY TYPE ===')
for t, m in r['by_type'].items():
    print('  %-8s(%3d): MM=%3.0f%%  TO=%3.0f%%  Open=%3.0f%%' % (
        t, m['count'], m['mm_accuracy']*100, m['to_accuracy']*100, m['open_accuracy']*100))
print()
print('=== BY DIFFICULTY ===')
for d, m in r['by_difficulty'].items():
    print('  %-8s(%3d): MM=%3.0f%%  TO=%3.0f%%  Open=%3.0f%%' % (
        d, m['count'], m['mm_accuracy']*100, m['to_accuracy']*100, m['open_accuracy']*100))
print()
v  = r['visual_questions']
nv = r['non_visual_questions']
print('=== VISUAL vs NON-VISUAL ===')
print('  Visual   (%3d): MM=%3.0f%%  TO=%3.0f%%  Open=%3.0f%%' % (
    v['count'],  v['mm_accuracy']*100,  v['to_accuracy']*100,  v['open_accuracy']*100))
print('  NonVisual(%3d): MM=%3.0f%%  TO=%3.0f%%  Open=%3.0f%%' % (
    nv['count'], nv['mm_accuracy']*100, nv['to_accuracy']*100, nv['open_accuracy']*100))
print()
lat = r['latency']
print('=== LATENCY ===')
print('  MM=%.1fs  TO=%.1fs  Open=%.1fs  overhead=%.1fs  avg_img=%.2f' % (
    lat['mm_avg_sec'], lat['to_avg_sec'], lat['open_avg_sec'],
    lat['mm_overhead_vs_to_sec'], lat['avg_images_per_query']))
