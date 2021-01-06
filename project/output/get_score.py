#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import sys
import os

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
	score1 = 0
	score2 = 0
	score = 0.0
	cm = [[0, 0, 0, 0],
		  [0, 0, 0, 0],
		  [0, 0, 0, 0],
		  [0, 0, 0, 0]]

	for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
		g_stance, t_stance = g, t
		if g_stance == t_stance:
			score += 0.25
			score1 += 1
			if g_stance != 'unrelated':
				score += 0.50
				score1 -= 1
				score2 += 1
		if g_stance in RELATED and t_stance in RELATED:
			score += 0.25
			score1 += 1

		cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

	return score, cm, score1, score2

def print_confusion_matrix(cm):
	lines = []
	header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
	line_len = len(header)
	lines.append("-"*line_len)
	lines.append(header)
	lines.append("-"*line_len)

	hit = 0
	total = 0
	for i, row in enumerate(cm):
		hit += row[i]
		total += sum(row)
		lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
																   *row))
		lines.append("-"*line_len)
	print('\n'.join(lines))


def report_score(actual,predicted):
	score,cm, score1, score2 = score_submission(actual,predicted)
	best_score, _, s1, s2 = score_submission(actual,actual)

	print_confusion_matrix(cm)
	print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
	return score*100/best_score, score1, score2, s1, s2

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Not enough arguments.")
		exit(0)
	method_name = sys.argv[1]
	set_name = sys.argv[2]
	preds = []
	actuals = []
	with open(os.path.join("output", method_name + "_" + set_name + "_preds.txt"), "r") as r:
		preds = r.read().splitlines() 
	with open(os.path.join("reference", method_name + "_" + set_name + "_actuals.txt"), "r") as r:
		actuals = r.read().splitlines() 
	_, score1, score2, s1, s2 = report_score(actuals, preds)
	print("Score1: " +str(score1) + " out of " + str(s1) + "\t("+str(score1*100/s1) + "%)")
	print("Score2: " +str(score2) + " out of " + str(s2) + "\t("+str(score2*100/s2) + "%)")

	