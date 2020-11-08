from __future__ import division
import sympy as sp
import numpy as np
import time

"""
The goal of this program is to compute counts of pseudoholomorphic curves in the symplectic cobordism E(c,d) minus E(a,b), based on the recursive formula in the paper "Computing Higher Symplectic Capacities I". More precisely, we assume that a,b,c,d are integers, and work with the irrational ellipsoids E(c,d+eps) and delta * E(a,b+eps), with eps and delta sufficiently small. Here eps is a slight perturbation to make the ellipsoids irrational (to keep the dynamics nondegenerate) and delta is a sufficiently small shrinking factor so that delta * E(a,b+eps) sits inside of E(c,d).

Note that only the ratios b/a and d/c play any role in the recursion, and we will generally denote these by x.

In particular, in the case that c = d = 1 and b >> a, i.e. E(c,d) is a slightly perturbed ball and E(c,d) is a very skinyn ellipsoid, we record the numbers T_d of degree d curves in CP^2 with a full index local tangency constraint at a point which were first computed in "Counting curves with local tangency constraints" (i.e. T_1 = 1, T_2 = 1, T_3 = 4, T_4 = 26, and so on).

Recall that psi is an L-infinity homomorphism. The domain (denoted by V_{a,b}^can in CHSCI) is the space with basis the symbols A_1,A_2,A_3,..., or alternatively we can identify A_q with the Reeb orbit in the boundary of E(1,x) of qth smallest CZ index. The codomain (denoted by V_{a,b} in CHSCI) of phi is the space with basis beta_{(i,j)} for i and j nonnegative integers, not both zero (and also generators alpha_{(i,j)} with i,j positive integers, but these do not explicitly appear in this algorithm). We will typically denote a pair (i,j) by bb and a list of such pairs by bb_list.
"""


#Some user parameters:
use_memory = True #If true, we perform the recursion by storing all previously encountered values in memory, which tends to be faster. By default should be set to True, but the user can turn this off if memory is limited.
rat_comp = 'rational' #If 'rational', we perform all computations exactly. If 'float32' or 'float64, perform all computations with floating point precision. By default this should be set to 'rational', but the user can try 'float32' to speed up the computation.


#Have the option of using either exact rational computations using sympy's Rational functionality, or else 32-bit or 64-bit floating point accuracy. If time is not a bottleneck it's best to use rational since this gives rigorous computations, but float32 should suffice for practice purposes.
def frac(p,q,rat_comp):
	if rat_comp == 'rational':
		return sp.Rational(p,q)
	elif rat_comp == 'float64':
		return np.float64(p/q)
	elif rat_comp == 'float32':
		return np.float32(p/q)
	else:
		raise Exception('Error! Unknown value for rat_comp.')

fact = sp.factorial
gcd = sp.gcd

def min_gen(q,x): #Gives the pair (i,j) with i+j = q such that max(i,j*x) is minimal. This corresponds to (i^{a,b}(q),j^{a,b}(q)) in the notation of CHSCI, with x = b/a.
	out = None
	out_act = None
	for i in range(q+1):
		j = q - i
		act = max(i,j*x)
		if (out == None) or (act < out_act) or (act == out_act and j < out[1]): #If j is smaller, than we consider the action smaller since we're really thinking of the ellipsoid as being E(1,x+epsilon).
			out = (i,j)
			out_act = act
	return out


def min_gen_as_orbit(q,x): #Gives the name of the orbit of qth smallest CZ index in the boundary of E(a,b) for x = b/a. Output: a pair, e.g. ('short',5), which represents the five-fold cover of the short simple Reeb orbit. 
	out_act = None
	orb_type = None
	orb_mult = None
	out_pair = None
	for i in range(q+1):
		j = q - i
		act = max(i,j*x)
		if (out_act == None) or (act < out_act) or (act == out_act and j < out_pair[1]): #If j is smaller, than we consider the action smaller since we're really thinking of the ellipsoid as being E(1,x+epsilon).
			out_act = act
			out_pair = (i,j)
			if i > j*x:
				orb_type = 'short'
				orb_mult = i
			else:
				orb_type = 'long'
				orb_mult = j
	return orb_type,orb_mult


def is_min_gen(bb,x): #Here bb is a pair (i,j), and we return true if (i,j) is the generator of minimal action with q = i + j. Recall that the action is given by max(i,j*x).
	i,j = bb
	q = i + j
	if bb == min_gen(q,x):
		return True
	else:
		return False

def brac_coeff(i,j,k,l,rat_comp): #The determinant coefficient appearing in front of the bracket.
	i,j,k,l = frac(i,1,rat_comp),frac(j,1,rat_comp),frac(k,1,rat_comp),frac(l,1,rat_comp)
	return frac(i*l-j*k,1,rat_comp)

#If use_memory is true, we store all previously computed values of phi in memory in order to save computation. The keys are of the form (x,bb_tuple), where bb_tuple is a tuple of pairs (i,j), assumed to be sorted according to Python's default sorting key.
if use_memory:
	stored_phi_values = {}


def compute_C_factor(q,x,rat_comp): #This is the constant C_{q;a,b}, which depends only on q and x = b/a. The precise value giving enumerative invariants is described in Theorem 5.3.2 in CHSCI.
	i_min,j_min = min_gen(q,x)
	num = gcd(i_min,j_min)
	denom = gcd(gcd(i_min,j_min),min_gen_as_orbit(i_min+j_min,x)[1])
	return frac(num,denom,rat_comp)

def compute_phi(bb_list,x,rat_comp,use_memory): #This is the main routine, which computes phi^k applied to k inputs which are beta basis elements. Here bb_set is a list of tuples (i,j). Note that tuples can be repeated, but their ordering is immaterial (and we don't have to worry about signs since they're all even degree). Output: a single number, the coefficient of gamma_q in phi^k(bb_list), where gamma_q is the unique basis element having the same the degree as phi_k(bb_list). Here k is the length of bb_list and phi^k is the part of the L-infinity homomorphism phi with k inputs. See Construction 3.2.1 in CHSCI for details of the recursion.
	if use_memory:
		bb_tuple = tuple(sorted(bb_list)) #In order to look it up, we first need to convert bb_tuple to a canonical, hashable form.
		if (x,bb_tuple) in stored_phi_values: #If we've already computed compute_phi for this input we simply return the stored value.
			return stored_phi_values[(x,bb_tuple)]

	#Otherwise, we have to compute it:

	assert len(bb_list) >= 1
	if len(bb_list) == 1:
		i,j = bb_list[0]
		q = i+j
		i_min,j_min = min_gen(q,x)		

		num = fact(i_min)*fact(j_min)
		C_const = compute_C_factor(q,x,rat_comp)
		den = fact(i)*fact(j)*C_const

		out = frac(num,den,rat_comp)

		if use_memory:
			stored_phi_values[(x,bb_tuple)] = out
		return out
	else:
		non_min_word_index = None
		for ctr in range(len(bb_list)):
			bb = bb_list[ctr]
			if is_min_gen(bb,x) == False:
				non_min_word_index = ctr
		if non_min_word_index == None: #In this case every generator in bb_list is minimal, so the output is zero.
			out = frac(0,1,rat_comp)
			if use_memory:
				stored_phi_values[(x,bb_tuple)] = out			
			return out

		#If we've gotten to here, there's a non-minimal word. We apply the main recursive step to bring it closer to a minimal word.
		bb = bb_list[non_min_word_index]
		i_non_min,j_non_min = bb
		i_min,j_min = min_gen(i_non_min+j_non_min,x)
		bb_list_rem = bb_list[:non_min_word_index] + bb_list[non_min_word_index+1:]

		if i_non_min < i_min:
			out = frac(0,1,rat_comp)
			out += frac(i_non_min+1,j_non_min,rat_comp)*compute_phi([(i_non_min+1,j_non_min-1)] + bb_list_rem,x,rat_comp,use_memory)
			for ctr in range(len(bb_list_rem)):
				bb_ctr = bb_list_rem[ctr]
				k,l = bb_ctr
				bb_list_rem_rem = bb_list_rem[:ctr] + bb_list_rem[ctr+1:]
				out -= frac(1,j_non_min,rat_comp)*brac_coeff(i_non_min+1,j_non_min,k,l,rat_comp)*compute_phi([(i_non_min+1+k,j_non_min+l)] + bb_list_rem_rem,x,rat_comp,use_memory)
			if use_memory:
				stored_phi_values[(x,bb_tuple)] = out			
			return out
		elif i_non_min > i_min:
			out = frac(0,1,rat_comp)
			out += frac(j_non_min+1,i_non_min,rat_comp)*compute_phi([(i_non_min-1,j_non_min+1)] + bb_list_rem,x,rat_comp,use_memory) 
			for ctr in range(len(bb_list_rem)):
				bb_ctr = bb_list_rem[ctr]
				k,l = bb_ctr				
				bb_list_rem_rem = bb_list_rem[:ctr] + bb_list_rem[ctr+1:]
				out += frac(1,i_non_min,rat_comp)*brac_coeff(i_non_min,j_non_min+1,k,l,rat_comp) * compute_phi([(i_non_min+k,j_non_min+1+l)] + bb_list_rem_rem,x,rat_comp,use_memory) #Note the negative sign here instead of positive sign!	
			if use_memory:	
				stored_phi_values[(x,bb_tuple)] = out				
			return out
		else:
			raise Exception


def compute_psi(A_list,x,rat_comp): #Input: A_list = [q_1,...,q_k] is a list of numbers. Output: (bb_list,total_coeff), where bb_list = [(i_1,j_1),...,(i_k,j_k)] corresponds to a list of beta generators, and total_coeff is the total coefficient sitting in front of it. See Construction 3.2.1 in CHSCI for details of the recursion.
	bb_list = []
	total_coeff = 1
	for q in A_list:
		bb_list.append(min_gen(q,x))
		total_coeff *= compute_C_factor(q,x,rat_comp)
	return bb_list,total_coeff

def compute_mu_factor(A_list): #Computes the factor mu which is related to the number of ways of ordering the elements of A_list.  
	out = 1
	for elt in set(A_list):
		out *= fact(A_list.count(elt))
	return out

def compute_kappa_factor(A_list,x): #Computes the factor kappa which is the product of the multiplicities of the corresponding orbits.
	out = 1
	for q in A_list:
		out *= min_gen_as_orbit(q,x)[1] #Recall: the output of min_gen_as_orbit is e.g. ('short',4), corresponding to the 4-fold cover of the short orbit.
	return out

def convert_short_k_to_A_q(k,x,rat_comp): #Input: k, corresponding to the k-fold cover of the short orbit in the boundary of E(1,x). Output: q, such that short^k is the orbit of qth smallest CZ index.
	out = k
	i = 1
	while(i*x < k):
		out += 1
		i += 1
	return out

def convert_long_k_to_A_q(k,x,rat_comp): #Input: k, corresponding to the k-fold cover of the long orbit in the boundary of E(1,x). Output: q, such that long^k is the orbit of qth smallest CZ index.
	out = k
	i = 1
	while(i <= k*x):
		out += 1
		i += 1
	return out

# The user interface:
print('Counting rational curves in E(c,d+eps) minus delta * E(a,b+eps) with several positive ends and one negative end.\n\n DISCLAIMER: not all of these counts are J-invariant (see e.g. section 5.2 in CHSCI), and these counts might be wrong if some of the curves could be multiply covered (see Rmk 5.2.1 in CHSCI).\n\n Please enter the following values:')
a = input('a: ')
b = input('b: ')
c = input('c: ')
d = input('d: ')
alpha = input('Please enter the ends asymptotic to multiples of the short orbit. For example, entering 1,1,5 corresponds to one end asymptotic to the short orbit, one end asymptotic to the double cover of the short orbit, and five ends asymptotic to the triple cover of the short orbit. Press enter if no short top ends.\n')
if alpha == '':
	alpha = []
else:
	alpha = [int(elt) for elt in alpha.split(',')]
beta = input('Similarly, please enter the ends asymptotic to multiples of the long orbit. For example, entering 1,1,5 corresponds to one end asymptotic to the long orbit, one end asymptotic to the double cover of the long orbit, and five ends asymptotic to the triple cover of the long orbit. Please enter if no long top ends.\n')
if beta == '':
	beta = []
else:
	beta = [int(elt) for elt in beta.split(',')]

x = frac(b,a,rat_comp)
y = frac(d,c,rat_comp)

A_list = []
for i in range(len(alpha)): #Convert the short orbits input generators A_q.
	q = convert_short_k_to_A_q(i+1,y,rat_comp)
	A_list.extend([q]*alpha[i])
for i in range(len(beta)): #Convert the long orbits input generators A_q.
	q = convert_long_k_to_A_q(i+1,y,rat_comp)
	A_list.extend([q]*beta[i])


print('Top ends:')
print([min_gen_as_orbit(q,y) for q in A_list])
print('Bottom end:')

tic = time.time()

q_bot = sum(A_list) + len(A_list) - 1
print(min_gen_as_orbit(q_bot,x))

#According to Theorem 5.3.2. of CHSCI, the count is obtained by applying psi for y and then phi for x:
bb_list,total_coeff = compute_psi(A_list,y,rat_comp)

out = total_coeff * compute_phi(bb_list,x,rat_comp,use_memory)

kappa_bottom = compute_kappa_factor([q_bot],x)
mu_top = compute_mu_factor(A_list)

out *= frac(1,kappa_bottom,rat_comp) * frac(1,mu_top,rat_comp)

print('Count: %s' %out)

toc = time.time()
print('total time elapsed: %s seconds' %(toc-tic))


