import numpy, itertools, random

# Simplified Version of Blackjack

CARDS = [1,2,3,4]
TGT = 10

def play(agent):
	v = 0
	running = True
	while running:
		last = random.choice (CARDS)
		v += last 
		if v > TGT:
			return 0
		running = agent.act(last)
	return float(v) / TGT

def playAgainst(other, agent):
	ys = play(agent)
	ts = play(other)

	if ys > ts:
		return 1.
	elif ts > ys:
		return 0.
	else:
		return .5



class AiAgent:
	def __init__(self, rnn):
		self.rnn = rnn
		self.inst = self.rnn.createInstance()


	def act(self, v):
		o = self.inst.step([v])
		return o[0] >= 0.5

class RuleBasedAgent:
	def __init__(self):
		self.v = 0

	def act(self, v):
		self.v += v
		return self.v < 8

class HumanAgent:
	def act(self, v):
		print "Last Card: " + str(v)
		print "Twist? [yn]"

		return raw_input() == "y"


class Trainer:
	runs = 100
	pop = 20
	breed = 5

	def __init__(self, play, generate):
		self.play = play
		self.generate = generate

	def evaluate(self, agent):
		s = 0.
		for i in range(self.runs):
			s += self.play(agent)
		return s / self.runs

	def evaluatePop(self, pop):
		return map(self.evaluate, pop)

	def train(self, e, n, pop=None, gen=1):
		if pop is None:
			pop = []

		# Generate new agents
		if len(pop) < self.pop:
			pop += [self.generate() for i in range(self.pop - len(pop))]

		scores = self.evaluatePop(pop)
		popScores = sorted(zip(scores, pop), reverse = True)
		best = popScores[0][0]
		
		pop = map(lambda x: x[1], popScores[0:self.breed])

		print "Generation #" + str(gen) + " - Best Score: " + str(best)

		if gen >= n:
			return pop
		if best >= e:
			return pop

		culled = map(lambda x: x[1], popScores[self.breed:])
		culled = map(lambda a: a.randomize() or a, culled)

		# Children learn from their parents through BPTT, on
		# random sample data
		parents = itertools.combinations(pop, 2)
		parentsAndChild = zip(itertools.cycle(parents), culled)
		for (pa,pb), c in parentsAndChild:
			u = numpy.random.uniform(size=(100, c.nin), low=-1., high=1.)
    		c.train(u, pb.run(u, numpy.zeros(c.n))[1], numpy.zeros(c.n), 0.25)
    		c.train(u, pa.run(u, numpy.zeros(c.n))[1], numpy.zeros(c.n), 0.25)


		pop = pop + culled

		return self.train(e, n, pop, gen + 1)


if __name__ == "__main__":
	import sys
	
	from rnn import RNN
	import pickle

	pop = None
	try:
		with open('test2', 'r') as f:
			pop = pickle.load(f)
		print "Loaded population"
	except Exception:
		pass


	if len(sys.argv) > 1 and sys.argv[1] == "train":
		

		trainer = Trainer(
			#play = lambda rnn: play(AiAgent(rnn)),
			play = lambda rnn: playAgainst(RuleBasedAgent(), AiAgent(rnn)),
			generate = lambda: RNN(20, 1, 1).randomize()
		)

		pop = trainer.train(0.9, 250, pop)

		with open('test2', 'w') as f:
			pickle.dump(pop, f)

	else:
		running = True
		while running:
			ys = play(RuleBasedAgent())
			print "Your Score: " + str(ys)

			ts = play(AiAgent(pop[0]))
			print "AI Score: " + str(ts)

			if ys > ts:
				print "You Win!"
			elif ts > ys:
				print "AI Wins!"
			else:
				print "Draw"


