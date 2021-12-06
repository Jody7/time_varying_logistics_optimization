from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

import math, random, collections
import subprocess, json
from multiprocessing.connection import Client

subprocess.Popen("python graph.py")
address = ('localhost', 6000)
conn = Client(address, authkey=bytes('graph', 'utf-8'))


m = ConcreteModel()
m.t = ContinuousSet(bounds=(0, 120))
nfe_count = 120
m.const_list = ConstraintList()

class AluminumFactory():
	@property
	def id(self):
		return alum_factory_list.index(self)

	def __init__(self, maxSize, prodRate, outflowRate, toList):
		self.maxSize = maxSize
		self.prodRate = prodRate
		self.outflowRate = outflowRate
		self.toList = toList
	def str(self):
		return "alum_factory%s" % self.id

	@property
	def stateVar(self):
		return m.component("alum_factory%s" % self.id)
	@property
	def dotVar(self):
		return m.component("alum_factory_dot%s" % self.id)
	@property
	def prodBool(self):
		return m.component("alum_factory_prod%s" % self.id)

class SodaFactory():
	@property
	def id(self):
		return soda_factory_list.index(self)

	def __init__(self, maxSize, outflowRate, toList):
		self.maxSize = maxSize
		self.outflowRate = outflowRate
		self.toList = toList
	def str(self):
		return "soda_factory%s" % self.id

	@property
	def stateVar(self):
		return m.component("soda_factory%s" % self.id)
	@property
	def dotVar(self):
		return m.component("soda_factory_dot%s" % self.id)

class CustomerStore():
	@property
	def id(self):
		return cust_store_list.index(self)

	def __init__(self, maxSize, prodRate):
		self.maxSize = maxSize
		self.prodRate = prodRate
	def str(self):
		return "cust_store%s" % self.id

	@property
	def stateVar(self):
		return m.component("cust_store%s" % self.id)
	@property
	def dotVar(self):
		return m.component("cust_store_dot%s" % self.id)
	@property
	def prodBool(self):
		return m.component("cust_store_prod%s" % self.id)

alum_factory_list = [AluminumFactory(100.0, 8.0, 10.0, [0, 1]), AluminumFactory(100.0, 8.0, 10.0, [1, 2])]
soda_factory_list = [SodaFactory(50.0, 5.0, [0, 1]), SodaFactory(50.0, 5.0, [0, 1]), SodaFactory(50.0, 5.0, [0, 1])]
#alum_factory_list = [AluminumFactory(100.0, 10.0, 10.0, [0])]
#soda_factory_list = [SodaFactory(100.0, 10.0, [0, 1])]
cust_store_list = [CustomerStore(30.0, -5.0), CustomerStore(30.0, -5.0)]

for factory in alum_factory_list:
	m.add_component("alum_factory%s" % factory.id, Var(m.t))
	m.add_component("alum_factory_dot%s" % factory.id, DerivativeVar(factory.stateVar))
	m.add_component("alum_factory_prod%s" % factory.id, Var(m.t, bounds=(0.0, 1.0)))

	m.const_list.add(factory.stateVar[0] == 0)
	m.add_component("alum_factory_state_min%s" % factory.id, Constraint(m.t, rule=lambda m, t, factory=factory: factory.stateVar[t] >= 0))
	m.add_component("alum_factory_state_max%s" % factory.id, Constraint(m.t, rule=lambda m, t, factory=factory: factory.stateVar[t] <= factory.maxSize))

for factory in soda_factory_list:
	m.add_component("soda_factory%s" % factory.id, Var(m.t))
	m.add_component("soda_factory_dot%s" % factory.id, DerivativeVar(factory.stateVar))

	m.const_list.add(factory.stateVar[0] == 0)
	m.add_component("soda_factory_state_min%s" % factory.id, Constraint(m.t, rule=lambda m, t, factory=factory: factory.stateVar[t] >= 0))
	m.add_component("soda_factory_state_max%s" % factory.id, Constraint(m.t, rule=lambda m, t, factory=factory: factory.stateVar[t] <= factory.maxSize))

for store in cust_store_list:
	m.add_component("cust_store%s" % store.id, Var(m.t))
	m.add_component("cust_store_dot%s" % store.id, DerivativeVar(store.stateVar))
	m.add_component("cust_store_prod%s" % store.id, Var(m.t, bounds=(0.0, 1.0)))

	m.const_list.add(store.stateVar[0] == 0)
	m.add_component("cust_store_state_min%s" % store.id, Constraint(m.t, rule=lambda m, t, store=store: store.stateVar[t] >= 0))
	m.add_component("cust_store_state_max%s" % store.id, Constraint(m.t, rule=lambda m, t, store=store: store.stateVar[t] <= store.maxSize))

# ----
soda_factory_eq = collections.defaultdict(list)

for a_factory in alum_factory_list:
	rate_var_list = []
	for s_factory_idx in a_factory.toList:
		s_factory = soda_factory_list[s_factory_idx]
		m.add_component("alum_factory%s_to_soda_factory%s" % (a_factory.id, s_factory.id), Var(m.t, bounds=(0, 1)))
		rate_var = m.component("alum_factory%s_to_soda_factory%s" % (a_factory.id, s_factory.id))
		rate_var_list.append(rate_var)

		soda_factory_eq[s_factory_idx].append((a_factory.outflowRate / len(a_factory.toList), rate_var))

	m.add_component("alum_factory%s_rateconst1" % (a_factory.id), Constraint(m.t, rule=lambda m, t, a_factory=a_factory, s_factory=s_factory, rate_var_list=rate_var_list: a_factory.dotVar[t] == (-sum([a_factory.outflowRate*rate_var[t] for rate_var in rate_var_list]) + a_factory.prodBool[t]*a_factory.prodRate)))
	m.add_component("alum_factory%s_rateconst2" % (a_factory.id), Constraint(m.t, rule=lambda m, t, a_factory=a_factory, s_factory=s_factory, rate_var_list=rate_var_list: sum([rate_var[t] for rate_var in rate_var_list]) <= 1.0))

cust_store_eq = collections.defaultdict(list)
for s_factory in soda_factory_list:
	rate_var_list = []
	for store_idx in s_factory.toList:
		store = cust_store_list[store_idx]
		m.add_component("soda_factory%s_to_cust_store%s" % (s_factory.id, store.id), Var(m.t, bounds=(0, 1)))
		rate_var = m.component("soda_factory%s_to_cust_store%s" % (s_factory.id, store.id))
		rate_var_list.append(rate_var)

		soda_factory_eq[s_factory_idx].append((-s_factory.outflowRate / len(s_factory.toList), rate_var))
		cust_store_eq[store_idx].append((s_factory.outflowRate / len(s_factory.toList), rate_var))

	#m.add_component("soda_factory%s_rateconst2" % (s_factory.id), Constraint(m.t, rule=lambda m, t, s_factory=s_factory, store=store, rate_var_list=rate_var_list: sum([rate_var[t] for rate_var in rate_var_list]) <= 1.0))

for store_idx in range(len(cust_store_list)):
	m.add_component("cust_store%s_rateconst1" % (store_idx), Constraint(m.t, rule=lambda m, t, store_idx=store_idx: cust_store_list[store_idx].dotVar[t] == (cust_store_list[store_idx].prodBool[t]*cust_store_list[store_idx].prodRate) + sum(x[0]*x[1][t] for x in cust_store_eq[store_idx])))
for s_factory_idx in range(len(soda_factory_list)):
	m.add_component("soda_factory%s_rateconst1" % (s_factory_idx), Constraint(m.t, rule=lambda m, t, s_factory_idx=s_factory_idx: soda_factory_list[s_factory_idx].dotVar[t] == sum(x[0]*x[1][t] for x in soda_factory_eq[s_factory_idx])))

def demand_target_func(t):
	#if t < 60:
	#	return 0
	return (sin(t*0.1 - math.pi/2)+1)/2

def cost_function(m, t):
	demand_target = demand_target_func(t)
	return (demand_target - m.cust_store_prod0[t])**2 + (demand_target - m.cust_store_prod1[t])**2

m.integral = Integral(m.t, wrt=m.t, rule=cost_function)
m.obj = Objective(expr=m.integral)

TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=nfe_count)
SolverFactory('gurobi').solve(m)

for t in m.t:
	print(m.alum_factory0[t]())
	print(m.soda_factory0[t]())
	print(m.cust_store0[t](), m.cust_store1[t]())
	print(m.cust_store_prod0[t]())
	#print("==")
	#print(m.alum_factory0_to_soda_factory0[t]())
	#print(m.soda_factory0_to_cust_store0[t]())
	print("---------")

def add_edges():
	graph_edges = []
	for a in alum_factory_list:
		for s_idx in a.toList:
			graph_edges.append([a.str(), soda_factory_list[s_idx].str()])
	for s in soda_factory_list:
		for c_idx in s.toList:
			graph_edges.append([s.str(), cust_store_list[c_idx].str()])
	return graph_edges

graph_edges = add_edges()

def flt_to_str(x):
	return str(round(x*100) / 100)

t_list = []
for t in m.t:
	t_list.append(t)

def get_node_from_string(node_name):
	node_obj = None
	node_list = []
	if "alum" in node_name:
		node_list = alum_factory_list
	elif "soda" in node_name:
		node_list = soda_factory_list
	elif "store" in node_name:
		node_list = cust_store_list
	for x in node_list:
		if x.str() == node_name:
			node_obj = x
			break
	return node_obj

def node_graph_string(num, node_name):
	node_obj = get_node_from_string(node_name)

	res = node_name + "\nAmt: " + flt_to_str(node_obj.stateVar[t_list[num]]())
	if "alum" in node_name:
		res += "\nProd Rate: " + flt_to_str(node_obj.prodRate * node_obj.prodBool[t_list[num]]())
	elif "cust" in node_name:
		res += "\nStore Demand: " + flt_to_str(node_obj.prodRate * node_obj.prodBool[t_list[num]]())
	return res

def graph_update(num, layout, G, ax):
	ax.clear()
	nx.draw(
		G, pos=layout, edge_color='black', width=1, linewidths=1,
		node_size=500, node_color='pink', alpha=0.9,
		labels={node: node_graph_string(num, node) for node in G.nodes()}
	)
	graph_edges_labels = {}
	for graph_edge_pair in graph_edges:
		edge_dt = m.component(graph_edge_pair[0] + "_to_" + graph_edge_pair[1])[t_list[num]]()
		node_obj = get_node_from_string(graph_edge_pair[0])
		graph_edges_labels[(graph_edge_pair[0], graph_edge_pair[1])] = flt_to_str(1/len(node_obj.toList) * edge_dt * get_node_from_string(graph_edge_pair[0]).outflowRate)

	nx.draw_networkx_edge_labels(
		G, pos=layout,
		edge_labels=graph_edges_labels,
		font_color='red'
	)
	ax.set_title("Frame {}".format(num))

	t_res = np.array([t for t in m.t]).tolist()
	graph_plot_data = {}
	graph_plot_data["demand_target"] = [5*demand_target_func(t) for t in m.t]

	graph_plot_data["cust_store_prod0"] = [5*m.cust_store_prod0[t]() for t in m.t]
	graph_plot_data["cust_store_prod1"] = [5*m.cust_store_prod1[t]() for t in m.t]

	for node_name in G.nodes():
		node_obj = get_node_from_string(node_name)
		graph_plot_data[node_name] = np.array([node_obj.stateVar[t]() for t in m.t]).tolist()

	conn.send(json.dumps([t_res, graph_plot_data]))

def graph_animation():
	fig, ax = plt.subplots(figsize=(10,10))

	edges = graph_edges
	G = nx.DiGraph()
	G.add_edges_from(edges)
	layout = nx.planar_layout(G)

	ani = animation.FuncAnimation(fig, graph_update, frames=nfe_count, fargs=(layout, G, ax))
	plt.show()

graph_animation()