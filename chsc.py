from __future__ import division
import sympy as sp
import numpy as np
import time

tic = time.time()
#We count curves in the cobordism with bottom bdy E(a,b+) and top bdy E(c,d+) (up to scaling). Here a,b,c,d should be integers. 
a = 10000
b = 1
c = 1
d = 1
top_degree = 11 #The higher degree curves we consider.
use_memory = True #If true, we perform the recursion by storing all previously encountered values in memory.
rat_comp = 'rational' #If 'rational', we perform all computations exactly. If 'float32', perform all computations with floating point precision.

"""
The goal of this program is to compute counts of pseudoholomorphic curves in the symplectic cobordism E(c,d) minus E(a,b), as given by the recursive formula in the paper Computing Higher Symplectic Capacities by Siegel. In particular, in the case that E(c,d) is a slightly perturbed ball and E(a,b) is a very skinny ellipsoid, we recover the numbers T_q of degree q curves in CP^2 with a full index local tangency constraint at a point (i.e. T_1 = 1, T_2 = 1, T_3 = 4, T_4 = 26, and so on).
"""

#Have the option of using either exact rational computations using sympy's Rational functionality, or else 32-bit or 64-bit floating point accuracy. If time is not a bottleneck it's best to use rational since this gives rigorous computations.
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

def min_gen(d,x): #Gives the generator bb_{i,j} with i+j = d such that max(i,j*x) is minimal. 
	out = None
	out_act = None
	for i in range(d+1):
		j = d - i
		act = max(i,j*x)
		if (out == None) or (act < out_act) or (act == out_act and j < out[1]): #If j is smaller, than we consider the action smaller since we're really thinking of the ellipsoid as being E(1,x+epsilon).
			out = (i,j)
			out_act = act
	return out


def min_gen_as_orbit(d,x):
	out_act = None
	orb_type = None
	orb_mult = None
	out_pair = None
	for i in range(d+1):
		j = d - i
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


def is_min_gen(bb,x): #Here bb is a pair (i,j), and we return true if (i,j) is the minimal generator in degree d = i + j.
	i,j = bb
	d = i + j
	if bb == min_gen(d,x):
		return True
	else:
		return False

def brac_coeff(i,j,k,l,rat_comp):
	i,j,k,l = frac(i,1,rat_comp),frac(j,1,rat_comp),frac(k,1,rat_comp),frac(l,1,rat_comp)
	return frac(i*l-j*k,1,rat_comp)

stored_phi_top_values = {}

def refresh_memory():
	stored_phi_top_values = {}

def phi_top(bb_list,x,rat_comp,use_memory): #Here bb_set is a list of tuples (i,j). Note that tuples can be repeated, but their ordering is immaterial (and we don't have to worry about signs since they're all even degree). Output: a single number, the coefficient of gamma_d in phi_k(bb_list), where k is the length of bb_list and gamma_d is the single element in the degree of phi_k(bb_list).
	if use_memory:
		bb_tuple = tuple(sorted(bb_list))
		if (x,bb_tuple) in stored_phi_top_values.keys(): #If we've already computed phi_top for this input we simply return the stored value.
			return stored_phi_top_values[(x,bb_tuple)]

	#Otherwise, we have to compute it:

	assert len(bb_list) >= 1
	if len(bb_list) == 1:
		i,j = bb_list[0]
		i_min,j_min = min_gen(i+j,x)
		# num = fact(i_min)*fact(j_min)
		num = fact(i_min)*fact(j_min)
		C_const = gcd(i_min,j_min)

		den = fact(i)*fact(j)*C_const


		out = frac(num,den,rat_comp)
		if use_memory:
			stored_phi_top_values[(x,bb_tuple)] = out
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
				stored_phi_top_values[(x,bb_tuple)] = out			
			return out

		#If we've gotten to here, there's a non-minimal word. We apply the main recursive step to bring it closer to a minimal word.
		bb = bb_list[non_min_word_index]
		i_non_min,j_non_min = bb
		i_min,j_min = min_gen(i_non_min+j_non_min,x)
		bb_list_rem = bb_list[:non_min_word_index] + bb_list[non_min_word_index+1:]

		if i_non_min < i_min:
			out = frac(0,1,rat_comp)
			out += frac(i_non_min+1,j_non_min,rat_comp)*phi_top([(i_non_min+1,j_non_min-1)] + bb_list_rem,x,rat_comp,use_memory)
			for ctr in range(len(bb_list_rem)):
				bb_ctr = bb_list_rem[ctr]
				k,l = bb_ctr
				bb_list_rem_rem = bb_list_rem[:ctr] + bb_list_rem[ctr+1:]
				out -= frac(1,j_non_min,rat_comp)*brac_coeff(i_non_min+1,j_non_min,k,l,rat_comp)*phi_top([(i_non_min+1+k,j_non_min+l)] + bb_list_rem_rem,x,rat_comp,use_memory)
			if use_memory:
				stored_phi_top_values[(x,bb_tuple)] = out			
			return out
		elif i_non_min > i_min:
			out = frac(0,1,rat_comp)
			out += frac(j_non_min+1,i_non_min,rat_comp)*phi_top([(i_non_min-1,j_non_min+1)] + bb_list_rem,x,rat_comp,use_memory) 
			for ctr in range(len(bb_list_rem)):
				bb_ctr = bb_list_rem[ctr]
				k,l = bb_ctr				
				bb_list_rem_rem = bb_list_rem[:ctr] + bb_list_rem[ctr+1:]
				out += frac(1,i_non_min,rat_comp)*brac_coeff(i_non_min,j_non_min+1,k,l,rat_comp) * phi_top([(i_non_min+k,j_non_min+1+l)] + bb_list_rem_rem,x,rat_comp,use_memory) #Note the negative sign here instead of positive sign!	
			if use_memory:	
				stored_phi_top_values[(x,bb_tuple)] = out				
			return out
		else:
			raise Exception



def orb_partitions(q,b=None): 
	if b == None:
		b = q
	else:
		assert b <= q
	if q == -1:
		return [[]]
	if q == 0:
		return []
	if q == 1:
		return [[1]]
	out = []
	for k in range(1,b+1):
		for elt in orb_partitions(q-k-1,min([k,b,q-k-1])):
			out.append([k]+elt)
	return out

def num_autos(list_input):
	out = 1
	for elt in set(list_input):
		out *= fact(list_input.count(elt))
	return out


top_ell = frac(c,d,rat_comp)
bottom_ell = frac(a,b,rat_comp)

x,y = top_ell,bottom_ell
print 'For cob with top E(1,%s+eps) and bottom E(1,%s+eps):' %(x,y)
print 
for q in range(1,top_degree+1):
	beta_word = [(1,1)]*q

	auto_factor = num_autos(beta_word)
	C_factors_from_psi = 1
	for elt in beta_word:
		C_factors_from_psi *= gcd(elt[0],elt[1])

	print 'degree: %s, # curves in CP^2 with local tangency constraint: %s' %(q,phi_top(beta_word,y,rat_comp,use_memory)*frac(C_factors_from_psi,1,rat_comp)/frac(auto_factor,1,rat_comp))
					

print
print '------------'
print


for q in range(1,top_degree+1):
	for op in orb_partitions(q):
		
		beta_word = [min_gen(i,x) for i in op]

		auto_factor = num_autos(beta_word)
		C_factors_from_psi = 1
		for elt in beta_word:
			C_factors_from_psi *= gcd(elt[0],elt[1])

		print 'top ends: %s, pred for # curves with one neg end: %s' %([min_gen_as_orbit(i,x) for i in op],phi_top(beta_word,y,rat_comp,use_memory)*frac(C_factors_from_psi,1,rat_comp)/frac(auto_factor,1,rat_comp))
						
	print '------'
print

toc = time.time()
print 'total time elapsed: %s seconds' %(toc-tic)