# Implementation of gamma-mixed coupling LP

import math
from optlang import Model, Variable, Constraint, Objective
from sympy import *
from itertools import combinations_with_replacement, combinations, product, chain

# parameters N_{max}, m^*, gamma from paper
NMAX = 7
mstar = 3
gamma = 25.597784

# list of LP constraints
constraints = []

# variables for flip probabilities
ps = list(symbols('p1:%d'%(NMAX+1)))
for i,x in enumerate(ps):
	if i == 1:
		var = Variable(x.name, lb=0, ub=1)	
	else:
		var = Variable(x.name, lb=0, ub=1)
	ps[i] = var

# one-indexing list of variables for flip probabilities
def p(i):
	if i in range(1,NMAX+1):
		return ps[i-1]
	elif i < 0:
		raise Exception("i too small")
	else:
		return 0

# p(1) = 1, flip probabilities nondecreasing, and constraint (10) from paper
constraints.append(Constraint(p(1), lb=1, ub=1))
for i in range(1,NMAX+1):
	constraints.append(Constraint(p(i) - p(i+1), lb=0))
	constraints.append(Constraint(p(i)*i, ub=1))

# lambda variables
lambda_bad = Variable('lambda_bad',lb = 1, ub = 2)
lambda_sing = Variable('lambda_sing',lb = 1, ub = 2)
lambda_good = Variable('lambda_good',lb = 1, ub = 2)

# picks out max entry and index of max entry in tuple l
def max_index(l):
	biggest = 0
	biggest_i = 0
	for i,x in enumerate(l):
		if x > biggest:
			biggest_i = i
			biggest = x
	return biggest_i, biggest

# all sorted l-tuples 0\le b_1\le\cdots\le b_l\le N_{max} which are not the all-zeros vector
def tuples(l):
	alltuples = list(combinations_with_replacement(range(0,NMAX+1),l))
	return [tuple(sorted(x)) for x in alltuples if sum(x) != 0]

# helper for tuple_pairs below
def shift(tup):
	if len(tup) == 1:
		return tup
	elif len(tup) == 2:
		return tuple([tup[1],tup[0]])
	else:
		return tuple([tup[1],tup[0]] + [tup[i] for i in range(2,len(tup))])

# all tuples (a^c,b^c) (modulo symmetries) from {0,...,NMAX}^l which are not the all-zeros vector
def tuple_pairs(l):
	alltuples = list(combinations_with_replacement(range(0,NMAX+1),l))
	sorted_tuples = [tuple(sorted(x,reverse=True)) for x in alltuples if sum(x) != 0]
	sorted_pairs = combinations_with_replacement(sorted_tuples,2)
	if l > 1:
		shifted_tuples = [shift(x) for x in sorted_tuples]
		sortshift_pairs = product(sorted_tuples,shifted_tuples)
		return chain(sorted_pairs,sortshift_pairs)
	else:
		return sorted_pairs

def upper(x):
	return min(x,NMAX+1)

# selects lambda based on (A,B,a_seq,b_seq)
def lam(A,B,a_seq,b_seq):
	if (a_seq == (1,1) and b_seq == (3,3) and A == 3 and B in [6,7]) or (b_seq == (1,1) and a_seq == (3,3) and A in [6,7] and B == 3):
		return lambda_bad
	elif len(a_seq) == 1:
		return lambda_sing
	else:
		return lambda_good

# memo for min_vars
memo = {}
# auxiliary variables corresponding to min(q_l,q'_l) terms
def min_vars(l):
	if len(l) == 4:
		s,t,u,v = l
		if s > u:
			u,v,s,t = s,t,u,v
			l = (s,t,u,v)
	if l in memo:
		return memo[l]
	else:
		if len(l) == 2:
			s,t = l
			s = upper(s); t = upper(t)
			if s >= NMAX + 1:
				memo[l] = 0
			elif t >= NMAX + 1:
				memo[l] = 0
			else:
				memo[l] = p(max(s,t))
		# ps and pt-pu
		elif len(l) == 3:
			s,t,u = l
			s = upper(s)
			t = upper(t)
			u = upper(u)
			if s >= NMAX + 1:
				memo[l] = 0
			elif t >= NMAX + 1:
				memo[l] = 0
			elif u <= t:
				raise Exception('NUUU')
			elif t >= s:
				memo[l] = p(s)
			elif u >= NMAX + 1:
				memo[l] = p(t)
			else:
				var = Variable('min%d-%d-%d'%(s,t,u), lb=-100, ub=1)
				memo[l] = var
				const1 = Constraint(p(s) - memo[l],lb=0)
				const2 = Constraint(p(t) - p(u) - memo[l],lb=0)
				constraints.append(const1); constraints.append(const2)
		# assume s <= u
		else:
			s,t,u,v = l
			s = upper(s)
			t = upper(t)
			u = upper(u)
			v = upper(v)
			if s >= NMAX + 1:
				memo[l] = 0
			elif t >= v:
				memo[l] = p(u) - p(v)
			elif t >= NMAX + 1:
				memo[l] = p(u) - p(v)
			elif v >= NMAX + 1:
				memo[l] = min_vars((u,s,t))
			else:
				var = Variable('min%d-%d-%d-%d'%(s,t,u,v), lb=0, ub=1)
				memo[l] = var
				const1 = Constraint(p(s) - p(t) - memo[l],lb=0)
				const2 = Constraint(p(u) - p(v) - memo[l],lb=0)
				constraints.append(const1); constraints.append(const2)
		return memo[l]

# constraint (11) for all A,B satisfying equation (2) in paper
def gen_eq(a_seq,b_seq):
	amax_i, amax = max_index(a_seq)
	bmax_i, bmax = max_index(b_seq)
	if amax_i != 0:
		raise Exception("bad seqs")
	if bmax_i not in [0,1]:
		raise Exception("bad seqs")
	if len(a_seq) != len(b_seq):
		raise Exception("bad seqs")
	l = len(a_seq)
	A_lower = amax + 1
	B_lower = bmax + 1
	A_upper = upper(sum(a_seq) + 1)
	B_upper = upper(sum(b_seq) + 1)
	# if i_max = j_max
	if bmax_i == 0:
		for A in range(A_lower,A_upper+1):
			for B in range(B_lower,B_upper+1):
				glob = lam(A,B,a_seq,b_seq)
				# compute H(A,B,a_seq,b_seq)
				f0 = amax*(p(amax) - p(A)) + bmax*(p(bmax) - p(B)) - min_vars((amax,A,bmax,B))
				fs = f0 + sum([a_seq[i]*p(a_seq[i]) + b_seq[i]*p(b_seq[i]) - p(max(a_seq[i],b_seq[i])) for i in range(1,l)])
				H = (l*glob - 1) - ((A - amax - 1)*p(A) + (B - bmax - 1)*p(B) + fs)
	# if i_max != j_max
	if bmax_i == 1:
		for A in range(A_lower,A_upper+1):
			for B in range(B_lower,B_upper+1):
				glob = lam(A,B,a_seq,b_seq)
				# compute H(A,B,a_seq,b_seq)
				f0 = amax*(p(amax) - p(A)) + b_seq[0]*(p(b_seq[0])) - min_vars((b_seq[0],amax,A))
				f1 = a_seq[1]*(p(a_seq[1])) - bmax*(p(bmax) - p(B)) - min_vars((a_seq[1],bmax,B))
				fs = f0 + f1 + sum([a_seq[i]*p(a_seq[i]) + b_seq[i]*p(b_seq[i]) - p(max(a_seq[i],b_seq[i])) for i in range(2,l)])
				H = (l*glob - 1) - ((A - amax - 1)*p(A) + (B - bmax - 1)*p(B) + fs)
	const = Constraint(H,lb=0)
	constraints.append(const)

# constraint (12) in paper
def gen_eq2(b_seq):
	if b_seq != tuple(sorted(b_seq)):
		raise EXCEPTION("bad seqs")
	B = sum(b_seq)
	l = len(b_seq)
	eq = (l*lambda_good - 1) - ((B - b_seq[-1])*p(B) + sum(b*p(b) for b in b_seq))
	const = Constraint(eq,lb=0)
	constraints.append(const)

# constraint (14) in paper
def approx(mstar):
	proxy1 = Variable('proxy1-%d' % mstar)
	proxy2 = Variable('proxy2-%d' % mstar)
	for A in range(0,NMAX + 2):
		eq = proxy1 - (A - 2)*p(A)
		const = Constraint(eq,lb=0)
		constraints.append(const)
	for a in range(0,NMAX+1):
		for b in range(a,NMAX+1):
			eq = proxy2 - a*p(a) - (b-1)*p(b)
			const = Constraint(eq,lb=0)
			constraints.append(const)
	eq = (mstar*lambda_good - 1) - (proxy1*2 + proxy2*mstar)
	const = Constraint(eq,lb=0)
	constraints.append(const)

# wrapper for gen_eq, gen_eq2, approx
def all_constraints(mstar):
	approx(mstar)
	for l in range(1,mstar):
		tup_pairs = tuple_pairs(l)
		for a_seq, b_seq in tup_pairs:
			gen_eq(a_seq,b_seq)
	for l in range(2,mstar):
		tups = tuples(l)
		for b_seq in tups:
			gen_eq2(b_seq)

# lamb is global lambda variable of the LP
lamb = Variable('lamb',lb=1,ub=2)
constraints.append(Constraint(lamb - lambda_bad*(gamma)/(gamma+1) - lambda_good*1/(gamma+1), lb = 0))
constraints.append(Constraint(lamb - lambda_sing, lb = 0))
constraints.append(Constraint(lamb - lambda_good, lb = 0))

obj = Objective(lamb, direction='min')

all_constraints(mstar)

# run LP
model = Model(name='vigoda')
model.objective = obj
model.add(constraints)
status = model.optimize()
print("status:", model.status)
print("objective value:", model.objective.value)
print("----------")
for var_name, var in model.variables.items():
    print(var_name, "=", var.primal)
