from __future__ import division
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from scipy.special import comb as choose
import sys
import time
from collections import Counter

"""
The purpose of this program is to compute counts of curves in CP^2 with various tangency constraints by recursion. We first implement the Gottsche-Pandharipande recursion for computing ordinary Gromov-Witten invariants of blowups of CP^2. We then implement the recursion which reduces tangency constraints to blowup Gromov-Witen invariants. Examples are given at the bottom.  

The data types we are using:
--> alpha is a tuple representing a linear combination of exceptional classes. For example, the tuple (4,4,4,3,1,1) represents the linear combination of 6 exceptional classes given by -4E_1-4E_2-4E_3-3E_4-E_5-E_6. 
We will often want to work with tuples which are in "canonical form", i.e. in decreasing order and have no 1's (every tuple can be made like this without changing the corresponding invariant).

--> C is a list of lists of the form [[1,2],[4,4,1]] etc, representing constraints with tangency conditions. For example, [[1,1],[2]] represents an ordinary double point at p_1 and a first order tangency constraint at p_2.
"""

def extend(list_of_lists,new_elements): #Given a list of lists, adds to each list one of several possibly new elements.
	new_list_of_lists = []
	for l in list_of_lists:
		for elt in new_elements:
			new_list_of_lists.append(l + [elt])
	return new_list_of_lists

def make_canonical_alpha(alpha): #Puts alpha in reverse order after discarding superfluous 1's.
	return tuple(sorted([elt for elt in alpha if elt > 1],reverse=True))

def decompositions(d_alpha): #Returns a list of tuples of the form (d1_alpha1,d2_alpha2), each representing a decomposition of d_alpha as in the R(m) recursion of Gottsche-Pandharipande.
	d,alpha = d_alpha
	out = []
	for d1 in range(1,d):
		d2 = d - d1
		alpha1_list = [[]]
		for i in range(len(alpha)):
			alpha1_list = extend(alpha1_list,range(np.max((0,alpha[i]-d2)),np.min((d1,alpha[i]))+1)) #The upper limits come from the fact that alpha1[i] is at most alpha[i], and it also cannot exceed d1, while the lower limits come from the fact that alpha2[i] = alpha[i]-alpha1[i] cannot exceed d2.
		for alpha1 in alpha1_list:
			d1_alpha1 = (d1,tuple(alpha1))
			alpha_arr = np.array(alpha)
			alpha2 = tuple(alpha_arr - np.array(alpha1))
			d2_alpha2 = (d2,alpha2)
			out.append((d1_alpha1,d2_alpha2))

	return out

gp_computed_values = {} #The previously computed values. We will only use keys d_alpha with alpha in canonical form.

#This function implements the Gottsche-Pandharipande recursion algorithm. The output is the number of curves of degree d in a blow-up of CP^2 in homology class whose exceptional part is specified by alpha, plus however many extra point constraints are needed to make the index 0.
def gp(d_alpha,verbose=False):
	global gp_computed_values

	d,alpha = d_alpha
	alpha_sum = np.sum(alpha)

	if len(alpha) > 0 and np.min(alpha) < 0:
		if d == 0 and len(alpha) == 1 and alpha_sum == -1:
			return 1
		else:
			return 0
	n = 3*d-1-alpha_sum
	if n < 0:
		return 0

	alpha = make_canonical_alpha(alpha)
	d_alpha = (d,alpha)
	alpha_sum = np.sum(alpha)
	n = 3*d-1-alpha_sum

	if d_alpha in gp_computed_values:
		return gp_computed_values[d_alpha]

	n = 3*d-1-alpha_sum #The number of extra point constraints needed to give the invariant corresponding to d_alpha index 0.

	if d == 1 and alpha_sum == 0:
		return 1

	if verbose:
		print('Applying GP recursion for d = %s, alpha = %s' %(d,str(alpha)))

	if np.sum([elt*(elt-1) for elt in alpha]) > (d-1)*(d-2): #This condition checks whether the invariant is automatically zero by the adjunction formula.
		gp_computed_values[d_alpha] = 0
		return 0

	if n >= 3:
		out = 0
		for d1_alpha1,d2_alpha2 in decompositions(d_alpha):
			d1,alpha1 = d1_alpha1
			d2,alpha2 = d2_alpha2

			n1 = 3*d1-1-np.sum(alpha1)
			out += gp(d1_alpha1,verbose)*gp(d2_alpha2,verbose)*(d1*d2-np.dot(alpha1,alpha2))*(d1*d2*choose(n-3,n1-1)-d1**2*choose(n-3,n1))
		gp_computed_values[d_alpha] = out
		return out

	elif len(alpha) > 0:
		a = alpha[0]
		alpha_decr = tuple([alpha[0]-1] + list(alpha[1:]))
		d_alpha_decr = (d,alpha_decr)

		out = (d**2 - (a-1)**2) * gp(d_alpha_decr,verbose)

		for d1_alpha1,d2_alpha2 in decompositions(d_alpha_decr):
			d1,alpha1 = d1_alpha1
			d2,alpha2 = d2_alpha2

			n1 = 3*d1-1-np.sum(alpha1)
			b = alpha1[0]
			c = alpha2[0]
			out += gp(d1_alpha1,verbose)*gp(d2_alpha2,verbose)*(d1*d2 - np.dot(alpha1,alpha2))*(d1*d2*b*c-d1**2*c**2)*choose(n,n1)	

		if out % (d**2*a) == 0:
			out = out // (d**2*a)
			gp_computed_values[d_alpha] = out
			return out

		else:
			raise Exception('Error! Expected divisibility in second branch of the recursion does not hold...')
	else:
		raise Exception('Error! Was not able to apply either branch of the recursion...')



def partitions_at_most_k(n,k): #Constructs an ordered list of partitions of n such that each part is at most k.
	if n == 0:
		return [[]]
	else:
		out = []
		for i in range(np.min([k,n]),0,-1):
			out = [[i]+elt for elt in partitions_at_most_k(n-i,i)] + out
		return out

partitions_computed_values = {} #We store already computed partitions in memory. Note that partitions are now taken to be tuples, and the values are lists of tuples.
def partitions(n): 
	global partitions_computed_values
	if n in partitions_computed_values:
		return partitions_computed_values[n]
	else:
		partitions_computed_values[n] = partitions_at_most_k(n,n)
		return partitions_computed_values[n]


def add_first_row_to_later_rows(P): #Given a partition P, constructs a list of partitions. The first element is given by leaving P alone, while the other elements are given by all possibly ways of removing the first row of P and adding it to a subsequent row (here thinking of partitions as Young diagrams with rows arrange in decreasing row from top to bottom).
	q = P[0]
	out = [P]
	for i in range(len(P)-1):
		new_P = P[1:]
		new_P[i] += q
		new_P.sort(reverse=True)
		out.append(new_P)
	return out


A_computed_values = {}

def A(k):
	global A_computed_values
	if k in A_computed_values:
		return A_computed_values[k]

	A = []
	parts = partitions(k)
	for i in range(len(parts)-1):
		A.append([add_first_row_to_later_rows(parts[i]).count(parts[j]) for j in range(1,len(parts))])
	A = np.array(A)
	A_computed_values[k] = A
	return A

A_inv_computed_values = {}

def A_inv(k):
	global A_inv_computed_values
	if k in A_inv_computed_values:
		return A_inv_computed_values[k]

	A_inv_computed_values[k] = inv(A(k))
	return A_inv_computed_values[k]

I_hat_computed_values = {}

def make_canonical_C(C):
	out  = [elt for elt in C if elt != [1]]
	out = tuple([tuple(elt) for elt in sorted(out)])
	return out

def I_hat(d,C,verbose=False): #This is the main function, which computes the tangency invariants for blowups of CP^2. Here d is the degree and C is a list of lists representing the constraints with tangency conditions. Here the hat refers to the fact that this counts curves with all marked points ordered. Below, the function I divides by the appropriate combinatorial factor to get the invariant corresponding to unordered marked points.
	C = [sorted(elt,reverse=True) for elt in C]

	d_C = (d,make_canonical_C(C))
	if d_C in I_hat_computed_values:
		return I_hat_computed_values[d_C]

	#We first check whether the invariant is automatically zero due to the adjunction inequality:
	lb = 0
	for P in C:
		for i in range(len(P)):
			for j in range(len(P)):
				lb += np.min([P[i],P[j]])
		lb -= np.sum(P)
	if lb > (d-1)*(d-2):
		I_hat_computed_values[d_C] = 0
		return 0

	if verbose:
		print('Applying tangency recursion with d = %s, C = %s' %(d,str(C)))

	# C = [elt for elt in C if elt != [1]]
	index_to_break_up = None
	for i in range(len(C)):
		if C[i] != [1]*len(C[i]):
			index_to_break_up = i
			break

	if index_to_break_up == None: #This means there are no tangency conditions to break up, so we can just apply the blow-up recursion.	
		alpha = make_canonical_alpha([np.sum(elt) for elt in C])
		d_alpha = (d,alpha)
		aut_factor = np.product([np.math.factorial(elt) for elt in alpha])
		out = aut_factor*gp(d_alpha,verbose)
		I_hat_computed_values[d_C] = out
		return out

	P = C[index_to_break_up]
	k = np.sum(P)
	q_p_partitions = partitions(k)[:-1]
	p_p_partitions = partitions(k)[1:]
	C_reduced = C[:index_to_break_up] + C[index_to_break_up+1:]
	C_q_p = [C_reduced + [[elt[0]]] + [elt[1:]] for elt in q_p_partitions]
	C_p_p = [C_reduced + [elt] for elt in p_p_partitions]

	v = np.array([I_hat(d,C) for C in C_q_p])
	w1 = I_hat(d,C_reduced + [[1]*k])
	w1_and_zeros = np.array([w1] + [0]*(len(v)-1))

	w = np.dot(A_inv(k),v-w1_and_zeros)

	P_index = p_p_partitions.index(P)

	out = w[P_index]

	I_hat_computed_values[d_C] = out
	return out

def combinatorial_factor(P):
	out = 1
	counts = Counter(P)
	for elt in counts:
		out *= np.math.factorial(counts[elt])
	return out

def I(d,C,verbose=False):
	cf = np.prod([combinatorial_factor(P) for P in make_canonical_C(C)])
	out = I_hat(d,C,verbose)
	if out % cf != 0:
		raise Exception('Error! The computed invariant %s is not divisible by the appropriate combinatorial factor %s.' %(out,cf))
	else:	
		return out // cf

def multinom(P):
	numer = np.math.factorial(np.sum(P))
	denom = np.prod([np.math.factorial(elt) for elt in P])
	if numer % denom != 0:
		raise Exception('Error! Did not get expected divisibility when computing multinomial coefficient.')
	out = numer // denom
	return out

# The user interface:
d = input('Please enter the degree of the tangency invariant in CP^2 you would like to compute.\n')
C = input('Now enter the tangency constraint. For example, [[3,2],[2]] signifies the constraint <T^2p_1,Tp_1,Tp_2,->, where "-" means that we add some number of extra point constraints to make the index zero.\n')
print I(d,C)

# A sample calculation (should get 34):
# print I(5,[[12,1,1]])
